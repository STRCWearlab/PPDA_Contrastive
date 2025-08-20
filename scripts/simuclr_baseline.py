import sys

sys.path.append("..")
import time
import argparse
import random
import json
import wandb
import pandas as pd
from dotenv import load_dotenv

from simuclr.datasets import SensorDataset, sliding_window
from simuclr.dataloaders import StratifiedSubsetSampler, FixedKPerClassSampler
from simuclr.utils import (
    get_default_device,
    seed_torch,
)
from simuclr.utils import paint

from models.utils import eval_model, wandb_logging

from scripts.script_utils import fine_tuning


def parse_range(range_str):
    try:
        # Split the input by comma and convert to a tuple of floats
        min_val, max_val = map(float, range_str.split(","))
        return (min_val, max_val)  # Return as a tuple
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Rotation range must be in the form 'min,max'."
        )


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="SimCLR Training for IMU-based HAR")

    # Dataset parameters
    parser.add_argument(
        "--dataset_name", type=str, default="realdisp", help="Name of dataset"
    )
    parser.add_argument(
        "--train_prefix",
        type=str,
        default="train",
        help="Prefix for training data",
    )
    parser.add_argument(
        "--val_prefix",
        type=str,
        default="val",
        help="Prefix for validation data",
    )
    parser.add_argument(
        "--test_prefix", type=str, default="test", help="Prefix for test data"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3], help="Random seeds"
    )
    parser.add_argument("--pt_encoder", type=str, default="TPN", help="Encoder model")

    # Fine-Tuning parameters
    parser.add_argument(
        "--ft_classifier", type=str, default="mlp", help="Classifier model. linear/mlp"
    )
    parser.add_argument("--ft_batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--ft_lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--ft_weight_decay", type=float, default=0.0, help="Weight decay"
    )
    parser.add_argument(
        "--ft_num_epochs", type=int, default=100, help="Number of training epochs"
    )
    # This should always be True for baseline setting
    parser.add_argument(
        "--ft_unfreeze",
        action="store_true",
        help="Unfreeze the encoder layers for fine-tuning",
    )
    parser.add_argument(
        "--ft_data_sample_type",
        type=str,
        default="stratified",
        help="Data sampling type",
    )
    parser.add_argument(
        "--ft_data_fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--ft_data_k_sample",
        type=int,
        default=1,
        help="k samples of data for each class to use for training",
    )
    parser.add_argument(
        "--ft_adjust_epochs",
        action="store_true",
        help="Adjust the number of epochs based on the fraction of data used",
    )
    parser.add_argument(
        "--ft_early_stopping_patience",
        type=int,
        default=100,
        help="Early stopping patience",
    )

    # WandB logging
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--offline", action="store_true", help="Run wandb offline")

    return parser.parse_args()


def configure_dataset_params(config):
    """
    Adjust parameters based on dataset selection.
    """
    dataset_defaults = {
        "realdisp": {
            "window": 100,
            "stride": 25,
            "n_channels": 12,
            "ft_batch_size": 256,
            "tpn_kernel_sizes": [24, 16, 8],
        },
        "realworld": {
            "window": 30,
            "stride": 15,
            "n_channels": 42,
            "ft_batch_size": 1024,
            "tpn_kernel_sizes": [12, 8, 4],
        },
        "mmfit": {
            "window": 60,
            "stride": 30,
            "n_channels": 18,
            "ft_batch_size": 128,
            "tpn_kernel_sizes": [24, 16, 8],
        },
    }

    if config.dataset_name in dataset_defaults:
        defaults = dataset_defaults[config.dataset_name]
        config.window = defaults["window"]
        config.stride = defaults["stride"]
        config.n_channels = defaults["n_channels"]
        config.ft_batch_size = defaults["ft_batch_size"]
        config.tpn_kernel_sizes = defaults["tpn_kernel_sizes"]
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset_name}")

    return config


if __name__ == "__main__":
    config = parse_args()
    configure_dataset_params(config)
    # Add path_processed to the config
    setattr(config, "path_processed", f"../data/{config.dataset_name}/")

    if config.wandb:
        WANDB_PROJECT = f"simuclr_{config.dataset_name}"
        # WANDB_ENTITY = "nobuyuki"
        randint = random.randint(0, 1000)
        WANDB_NAME = f"baseline_{randint}"

        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            mode="offline" if config.offline else "online",
            config=vars(config),
        )
        print(f"WandB initialized: {WANDB_PROJECT}/{WANDB_NAME}")

    # Get the default device (use cuda if available)
    device = get_default_device()

    # Prepare Train, Val, Test dataset
    dataset_train = SensorDataset(
        config.dataset_name,
        window=config.window,
        stride=config.stride,
        prefix=config.train_prefix,
        path_processed=config.path_processed,
        lazy_load=True,
        device=device,
    )
    # Use the mean and std from the training dataset
    scaling_config = {"mean": dataset_train.mean, "std": dataset_train.std}
    dataset_val = SensorDataset(
        config.dataset_name,
        window=config.window,
        stride=config.stride,
        prefix=config.val_prefix,
        path_processed=config.path_processed,
        lazy_load=True,
        device=device,
        **scaling_config,
    )
    dataset_test = SensorDataset(
        config.dataset_name,
        window=config.window,
        stride=config.stride,
        prefix=config.test_prefix,
        path_processed=config.path_processed,
        lazy_load=True,
        device=device,
        **scaling_config,
    )

    print(paint(f"Applied Settings: "))
    print(json.dumps(vars(config), indent=2, default=str))

    train_results_list = []
    test_results_list = []
    start_time = time.time()
    for seed in config.seeds:
        print(paint(f"[*] Running with seed: {seed}"))
        # Init the weights again before nunning the code below.
        seed_torch(seed=seed)

        _, train_target_list = sliding_window(
            dataset_train.data,
            dataset_train.target,
            dataset_train.window,
            dataset_train.stride,
        )

        if config.ft_data_sample_type == "k_sample":
            # pick {ft_data_fraction} samples per class
            print(f"Using {config.ft_data_k_sample} samples from each class")
            custom_sampler = FixedKPerClassSampler(
                train_target_list,
                k=int(config.ft_data_k_sample),
                seed=seed,
            )
        elif config.ft_data_sample_type == "stratified":
            # Use the fraction of the dataset.
            print(f"Using {config.ft_data_fraction} fraction from each class")
            custom_sampler = StratifiedSubsetSampler(
                train_target_list, fraction=config.ft_data_fraction, seed=seed
            )
        else:
            print("No custom sampler provided. Using default setting.")
            custom_sampler = None

        # Fine-tuning
        # Prepare Classifier
        model_ft, (
            t_loss,
            t_acc,
            t_fm,
            t_fw,
            v_loss,
            v_acc,
            v_fm,
            v_fw,
            criterion,
        ) = fine_tuning(
            dataset_train,
            dataset_val,
            dataset_test,
            config,
            seed,
            path_checkpoints=None,
            checkpoint_name=None,  # Not needed when path_checkpoints is None
            device=device,
            custom_sampler=custom_sampler,
            run_name=WANDB_NAME,
            unfreeze=True,
        )

        loss_test, acc_test, fm_test, fw_test, elapsed = eval_model(
            model_ft, dataset_test, batch_size=256, seed=seed
        )

        # Save the results
        results_row = {
            "v_type": "split",
            "seed": seed,
            "sbj": -1,
            "t_loss": t_loss,
            "t_acc": t_acc,
            "t_fm": t_fm,
            "t_fw": t_fw,
            "v_loss": v_loss,
            "v_acc": v_acc,
            "v_fm": v_fm,
            "v_fw": v_fw,
        }

        tests_results_row = {
            "v_type": "split",
            "seed": seed,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_fm": fm_test,
            "test_fw": fw_test,
        }

        train_results_list.append(results_row)
        test_results_list.append(tests_results_row)

    # After the loop, convert lists of dictionaries to DataFrames
    train_results = pd.DataFrame(train_results_list)
    test_results = pd.DataFrame(test_results_list)

    wandb_logging(train_results, test_results, vars(config), epoch_name="ft_num_epochs")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
    print(paint("--" * 75, "blue"))
    wandb.finish()
