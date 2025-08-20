import os
import sys
import warnings

sys.path.append("..")
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
from lightly.loss import NTXentLoss

from simuclr.utils import (
    get_default_device,
    extract_representations,
    sample_equal_per_class,
)
from simuclr.utils import paint

from models.utils import train_model
from models.simclr import SimCLR, TPN, Classifier
from models.deep_conv_lstm import DeepConvLSTM
from models.attend_and_discriminate import AttendAndDiscriminate
from models.utils import save_model, init_weights


# Prepare SimCLR
def pre_train_simclr(
    dataset_train,
    dataset_val,
    train_loader,
    val_loader,
    config,
    seed,
    run_name=None,
    device=None,
    save_checkpoint_epochs=[],  # Save checkpoints at these epochs
):
    if device is None:
        device = get_default_device()

    if config.pt_encoder == "TPN":
        print("Using TPN as the encoder...")
        encoder = TPN(
            in_channels=config.n_channels,
            dataset=config.dataset_name,  # for logging
            experiment=f"pretrain_seed_{seed}",  # for logging
            timestamp=run_name,
            kernel_sizes=config.tpn_kernel_sizes,
        )
    elif config.pt_encoder == "DeepConvLSTM":
        print("Using DeepConvLSTM as the encoder...")
        encoder = DeepConvLSTM(
            n_channels=config.n_channels,
            n_classes=dataset_train.n_classes,  # not relevant for pre-training but for compatibility
            dataset=config.dataset_name,  # for logging
            experiment=f"pretrain_seed_{seed}",  # for logging
            timestamp=run_name,
            as_backbone=True,  # no logits output
        )
    elif config.pt_encoder == "AttendAndDiscriminate":
        print("Using AttendAndDiscriminate as the encoder...")
        encoder = AttendAndDiscriminate(
            input_dim=config.n_channels,
            num_class=dataset_train.n_classes,  # not relevant for pre-training but for compatibility
            dataset=config.dataset_name,  # for logging
            experiment=f"pretrain_seed_{seed}",  # for logging
            timestamp=run_name,
            as_backbone=True,  # no logits output
        )
    else:
        raise NotImplementedError("Invalid Encoder Model", config.pt_encoder)

    simclr = SimCLR(backbone=encoder, device=device)
    optimizer = torch.optim.SGD(
        simclr.parameters(),
        lr=config.pt_lr,
        weight_decay=config.pt_weight_decay,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.pt_num_epochs
    )
    criterion = NTXentLoss(temperature=0.1)
    init_weights(simclr, method="orthogonal")

    # Fine-tuning loop
    min_val_loss = np.inf
    early_stopping_counter = 0
    for epoch in range(config.pt_num_epochs):
        simclr.train()
        loss_epoch = 0.0
        for X_aug_1, X_aug_2, _ in tqdm(train_loader):
            # print("Loaded data", X_aug_1.device, X_aug_2.device)
            X_aug_1 = X_aug_1.to(device)
            X_aug_2 = X_aug_2.to(device)
            # print("Loaded data", X_aug_1.device, X_aug_2.device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out_1, out_2 = simclr(X_aug_1), simclr(X_aug_2)
                loss = criterion(out_1, out_2)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item() * X_aug_1.size(0)
        scheduler.step()
        loss_epoch /= len(dataset_train)

        # Validation
        simclr.eval()
        val_loss_epoch = 0.0
        for X_aug_1, X_aug_2, _ in val_loader:
            X_aug_1 = X_aug_1.to(device)
            X_aug_2 = X_aug_2.to(device)
            with torch.set_grad_enabled(False):
                out_1, out_2 = simclr(X_aug_1), simclr(X_aug_2)
                loss = criterion(out_1, out_2)
                val_loss_epoch += loss.item() * X_aug_1.size(0)
        val_loss_epoch /= len(dataset_val)
        print(
            f"Epoch {epoch + 1:3d}/{config.pt_num_epochs}, Train Loss: {loss_epoch:.5f}, Val Loss: {val_loss_epoch:.5f}"
        )

        if config.wandb:
            wandb.log(
                {
                    f"train_loss_{seed}": loss_epoch,
                    f"val_loss_{seed}": val_loss_epoch,
                    f"epoch_{seed}": epoch + 1,
                }
            )

        # Save the best model based on validation loss
        if val_loss_epoch < min_val_loss:
            early_stopping_counter = 0  # Reset early stopping counter
            print(
                paint(
                    f"[*] Saving checkpoint... ({min_val_loss}->{val_loss_epoch})",
                    "blue",
                )
            )
            min_val_loss = val_loss_epoch
            save_model(
                simclr.backbone,
                epoch,
                config=vars(config),
                best=True,
                filename=None,
            )
        else:
            early_stopping_counter += 1

        if epoch + 1 in save_checkpoint_epochs:
            print(
                paint(
                    f"[*] Saving checkpoint... (Epoch {epoch + 1})",
                    "blue",
                )
            )
            save_model(
                simclr.backbone,
                epoch + 1,
                config=vars(config),
                best=False,
                filename=None,
            )

    best_model_path = f"{simclr.backbone.path_checkpoints}/model_best.pth"
    simclr.load_pretrained_weights(best_model_path)
    return simclr


def log_representation(simclr, dataset_test, seed, device, n_sample=50):
    # Representation Analysis
    print("Loading the best model for representation analysis...")
    simclr.eval()

    print("Generating test data embeddings...")
    loader_test = DataLoader(
        dataset_test,
        batch_size=256,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
    )

    all_embeddings, all_labels = extract_representations(simclr, loader_test, device)
    sampled_embeddings, sampled_labels = sample_equal_per_class(
        all_embeddings, all_labels, num_samples_per_class=n_sample
    )
    table = wandb.Table(columns=["embedding", "label"])
    for i in range(len(sampled_embeddings)):
        table.add_data(sampled_embeddings[i].tolist(), sampled_labels[i])
    wandb.log({f"embeddings_{seed}": table})
    print(f"Embeddings (embedding_{seed}) logged to WandB")


def fine_tuning(
    dataset_train,
    dataset_val,
    dataset_test,
    config,
    seed,
    path_checkpoints=None,
    checkpoint_name="model_best.pth",
    device=None,
    custom_sampler=None,
    unfreeze=None,
    run_name=None,
):
    if unfreeze is None:
        unfreeze = config.ft_unfreeze

    if device is None:
        device = get_default_device()

    n_epochs = config.ft_num_epochs
    if config.ft_adjust_epochs:
        if custom_sampler is None:
            raise ValueError(
                "custom_sampler must be provided to adjust epochs based on the number of samples used."
            )
        print("ft_num_epochs adjusted: ", config.ft_num_epochs, end="")
        n_epochs = scale_epochs(
            config.ft_num_epochs,
            num_samples_used=len(custom_sampler),
            num_total_samples=len(dataset_train),
            max_epochs=10000,
        )
        print(" -> ", n_epochs)

    if config.pt_encoder == "TPN":
        print("Using TPN as the encoder for fine-tuning...")
        encoder = TPN(
            in_channels=config.n_channels,
            dataset=config.dataset_name,  # for logging
            experiment=f"finetune_seed_{seed}",  # for logging
            kernel_sizes=config.tpn_kernel_sizes,
        )
    elif config.pt_encoder == "DeepConvLSTM":
        print("Using DeepConvLSTM as the encoder for fine-tuning...")
        as_backbone = not unfreeze
        encoder = DeepConvLSTM(
            n_channels=config.n_channels,
            n_classes=dataset_train.n_classes,
            dataset=config.dataset_name,  # for logging
            experiment=f"finetune_seed_{seed}",  # for logging
            as_backbone=as_backbone,
        )
    elif config.pt_encoder == "AttendAndDiscriminate":
        print("Using AttendAndDiscriminate as the encoder...")
        as_backbone = not unfreeze  # if unfreeze, we want to train the whole model
        # also, classification head is added here.
        encoder = AttendAndDiscriminate(
            input_dim=config.n_channels,
            num_class=dataset_train.n_classes,  # not relevant for pre-training but for compatibility
            dataset=config.dataset_name,  # for logging
            experiment=f"finetune_seed_{seed}",
            as_backbone=as_backbone,
        )
    else:
        raise NotImplementedError("Invalid Encoder Model", config.pt_encoder)

    if config.pt_encoder in ["DeepConvLSTM", "AttendAndDiscriminate"] and unfreeze:
        model_ft = encoder  # encoder has the classification head when unfreeze == True
        warnings.warn(
            "Unfreezing the encoder. Checkpoint path may overlap across different runs."
        )
    else:
        model_ft = Classifier(
            backbone=encoder,
            dataset=config.dataset_name,
            num_classes=dataset_train.n_classes,
            device=device,
            classification_model=config.ft_classifier,
            experiment=f"{config.pt_encoder}/{run_name}",  # for logging
            timestamp=f"finetune_seed_{seed}",
        )
    model_ft.to(device)

    # Load the pre-trained model with the best validation loss
    if path_checkpoints is None:
        weights_init = "orthogonal"
    else:
        best_model_path = f"{path_checkpoints}/{checkpoint_name}"
        print(f"Loading the best model from {best_model_path}...")
        model_ft.load_pretrained_weights(best_model_path)
        weights_init = None

    if unfreeze or path_checkpoints is None:
        print(
            "unfreeze=True or path_checkpoints=None. Not freezing the encoder layers..."
        )
    else:
        if config.ft_classifier == "mlp" and config.pt_encoder == "TPN":
            # This is only valid for TPN model
            print("Freezing the encoder layers... (except for the last Conv layer)")
            model_ft.freeze_two_conv_layers()
        elif config.ft_classifier == "mlp" and config.pt_encoder == "DeepConvLSTM":
            print("Freezing the encoder layers... (except for the LSTM layers)")
            model_ft.freeze_conv_layers_keep_lstm()
        elif (
            config.ft_classifier == "mlp"
            and config.pt_encoder == "AttendAndDiscriminate"
        ):
            print(
                "Freezing the encoder layers... (* check the output for unfrozen layers)"
            )
            model_ft.freeze_attend_layers()

        elif config.ft_classifier == "mlp" and not config.pt_encoder == "TPN":
            print(f"You are using {config.pt_encoder} as the encoder")
            print("Freezing the all the encoder layers...")
            model_ft.freeze_encoder_layers()
        elif config.ft_classifier == "linear":
            model_ft.freeze_encoder_layers()
        else:
            raise ValueError("Invalid classification model")

    (
        t_loss,
        t_acc,
        t_fm,
        t_fw,
        v_loss,
        v_acc,
        v_fm,
        v_fw,
        criterion,
    ) = train_model(
        model_ft,
        dataset_train,
        dataset_val,
        dataset_test,
        weights_init=weights_init,
        class_weights=None,
        seed=seed,
        n_epochs=n_epochs,
        batch_size_train=config.ft_batch_size,
        custom_sampler=custom_sampler,
        early_stopping_patience=config.ft_early_stopping_patience,
    )

    return model_ft, (t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, criterion)


def scale_epochs(base_epochs, num_samples_used, num_total_samples, max_epochs=1000):
    """
    Automatically scale number of epochs based on how much data is used.
    """
    scale = num_total_samples / max(num_samples_used, 1)
    scaled_epochs = int(base_epochs * scale)
    return min(scaled_epochs, max_epochs)


def get_checkpoint_paths(config, seed):
    base_path = f"./logs/{config.pt_encoder}/{config.dataset_name}/pretrain_seed_{seed}/{config.pt_run_name}/checkpoints/"

    if config.use_best_checkpoint:
        best_checkpoint = os.path.join(base_path, "model_best.pth")
        if not os.path.exists(best_checkpoint):
            raise FileNotFoundError(f"Best checkpoint not found in {base_path}")
        return base_path, None

    elif config.checkpoint_epoch_pt is not None:
        specific_checkpoint = os.path.join(
            base_path, f"model_epoch_{config.checkpoint_epoch_pt}.pth"
        )
        if not os.path.exists(specific_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint for epoch {config.checkpoint_epoch_pt} not found in {base_path}"
            )
        return base_path, f"model_epoch_{config.checkpoint_epoch_pt}.pth"

    else:
        raise ValueError(
            "Either use_best_checkpoint or checkpoint_epoch_pt must be specified"
        )


def configure_dataset_params(config):
    """
    Adjust parameters based on dataset selection.
    """
    dataset_defaults = {
        "realdisp": {
            "window": 100,
            "stride": 25,
            "n_channels": 12,
            "tpn_kernel_sizes": [24, 16, 8],
            "ft_batch_size": 256,
        },
        "realworld": {
            "window": 30,
            "stride": 15,
            "n_channels": 42,
            "tpn_kernel_sizes": [12, 8, 4],
            "ft_batch_size": 1024,
        },
        "mmfit": {
            "window": 60,
            "stride": 30,
            "n_channels": 18,
            "tpn_kernel_sizes": [24, 16, 8],
            "ft_batch_size": 128,
        },
    }

    if config.dataset_name in dataset_defaults:
        defaults = dataset_defaults[config.dataset_name]
        config.window = defaults["window"]
        config.stride = defaults["stride"]
        config.n_channels = defaults["n_channels"]
        config.tpn_kernel_sizes = defaults["tpn_kernel_sizes"]
        config.ft_batch_size = defaults["ft_batch_size"]
        print(paint(f"ft_batch_size: {defaults['ft_batch_size']}"))
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset_name}")

    return config
