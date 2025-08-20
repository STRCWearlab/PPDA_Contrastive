import sys
import os

sys.path.append("..")
import time
import argparse
import random
import json

import torch
import wandb
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from simuclr.datasets import SensorDataset, sliding_window
from simuclr.dataloaders import StratifiedSubsetSampler, FixedKPerClassSampler
from simuclr.augmentations.aug_planner import ContrastiveAugPolicyPlanner
from simuclr.dataloaders import STDACLRDataLoader
from simuclr.utils import (
    get_default_device,
    seed_torch,
)
from simuclr.augmentations import stda
from simuclr.utils import paint

from models.utils import eval_model, wandb_logging

from scripts.script_utils import (
    pre_train_simclr,
    log_representation,
    fine_tuning,
    configure_dataset_params,
)


def parse_range(range_str):
    try:
        # Split the input by comma and convert to a tuple of floats
        min_val, max_val = map(float, range_str.split(","))
        return min_val, max_val  # Return as a tuple
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

    # Pre-Training parameters
    parser.add_argument("--pt_encoder", type=str, default="TPN", help="Encoder model")
    parser.add_argument("--pt_batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--pt_lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--pt_weight_decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--pt_num_epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--pt_patience", type=int, default=100, help="Patience for early stopping"
    )
    parser.add_argument(
        "--pt_ckpt_interval",
        type=int,
        default=-1,
        help="Epochs to save checkpoints",
    )
    # Fine-Tuning parameters
    parser.add_argument(
        "--ft_classifier", type=str, default="mlp", help="Classifier model. linear/mlp"
    )
    parser.add_argument("--ft_lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--ft_weight_decay", type=float, default=0.0, help="Weight decay"
    )
    parser.add_argument(
        "--ft_num_epochs", type=int, default=50, help="Number of training epochs"
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
        "--ft_unfreeze",
        action="store_true",
        help="Unfreeze the encoder layers for fine-tuning",
    )
    parser.add_argument(
        "--ft_early_stopping_patience",
        type=int,
        default=100,
        help="Early stopping patience",
    )

    # Data Augmentation parameters
    parser.add_argument(
        "--first_augs",
        type=str,
        nargs="+",
        default=["magscale", "timescale"],
        help="List of 1st data augmentations to apply",
    )
    parser.add_argument(
        "--second_augs",
        type=str,
        nargs="+",
        default=[],
        help="List of 2nd data augmentations to apply",
    )
    parser.add_argument(
        "--magscale_mu", type=float, default=1.0, help="Magnitude scaling mean"
    )
    parser.add_argument(
        "--magscale_sigma", type=float, default=0.1, help="Magnitude scaling sigma"
    )
    parser.add_argument(
        "--magwarp_sigma", type=float, default=0.2, help="Magnitude warping sigma"
    )
    parser.add_argument(
        "--magwarp_knot", type=int, default=2, help="Magnitude warping K knot"
    )
    parser.add_argument(
        "--timescale_min", type=float, default=0.8, help="Minimum time scaling factor"
    )
    parser.add_argument(
        "--timescale_max", type=float, default=1.2, help="Maximum time scaling factor"
    )
    # Actually, timewarp_sigma is not used in the code
    parser.add_argument(
        "--timewarp_sigma",
        type=float,
        default=0.2,
        help="Time warping sigma (not used)",
    )
    parser.add_argument(
        "--timewarp_knot", type=int, default=2, help="Time warping K knot"
    )
    parser.add_argument(
        "--timewarp_max_speed_ratio",
        type=float,
        default=2,
        help="Time warping max_speed_ratio",
    )
    parser.add_argument(
        "--rotation_range_x",
        type=parse_range,
        default=(-180.0, 180.0),
        help="Rotation range for x-axis in the form 'min,max'",
    )
    parser.add_argument(
        "--rotation_range_y",
        type=parse_range,
        default=(-180.0, 180.0),
        help="Rotation range for y-axis in the form 'min,max'",
    )
    parser.add_argument(
        "--rotation_range_z",
        type=parse_range,
        default=(-180.0, 180.0),
        help="Rotation range for z-axis in the form 'min,max'",
    )
    parser.add_argument(
        "--jitter_sigma",
        type=float,
        default=0.1,
        help="Noise standard deviation (sigma)",
    )

    # WandB logging
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--offline", action="store_true", help="Run wandb offline")

    return parser.parse_args()


def prepare_aug_function(aug_name: str, config, device):
    if aug_name == "magscale":
        mu = torch.tensor(
            config.magscale_mu,
            requires_grad=False,
        )  # device=device)
        log_sigma = torch.tensor(
            np.log(config.magscale_sigma),
            requires_grad=False,
            # device=device,
        )
        magscale = stda.MagnitudeScaling(mu, log_sigma)
        return magscale
    elif aug_name == "magwarp":
        sigma = config.magwarp_sigma
        knot = config.magwarp_knot
        magwarp = stda.MagnitudeWarping(sigma=sigma, knot=knot)  # device=device)
        return magwarp
    elif aug_name == "timescale":
        scale_min = config.timescale_min
        scale_max = config.timescale_max
        timescale = stda.TimeScaling(
            scale_factor_min=scale_min, scale_factor_max=scale_max  # , device=device
        )
        return timescale
    elif aug_name == "timewarp":
        sigma = config.timewarp_sigma
        knot = config.timewarp_knot
        max_speed_ratio = config.timewarp_max_speed_ratio
        timewarp = stda.TimeWarping(
            sigma=sigma,
            knot=knot,
            max_speed_ratio=max_speed_ratio,  # device=device
        )
        return timewarp
    elif aug_name == "rotation":
        rot_range_x = torch.tensor(
            config.rotation_range_x,
        )  # device=device)
        rot_range_y = torch.tensor(
            config.rotation_range_y,
        )  # device=device)
        rot_range_z = torch.tensor(
            config.rotation_range_z,
        )  # device=device)
        rotation = stda.Rotation(
            rot_range_x,
            rot_range_y,
            rot_range_z,
        )  # device=device)
        return rotation
    elif aug_name == "jitter":
        log_sigma = torch.tensor(
            np.log(config.jitter_sigma),
            requires_grad=False,
            # device=device,
        )
        jitter = stda.Jittering(
            log_sigma,
        )  # device=device)
        return jitter
    else:
        raise NotImplementedError("Invalid Data Augmentation name", aug_name)


if __name__ == "__main__":
    load_dotenv()
    config = parse_args()
    configure_dataset_params(config)
    # Add path_processed to the config
    setattr(config, "path_processed", f"../data/{config.dataset_name}/")

    if config.pt_ckpt_interval == -1:
        config.save_checkpoint_epochs = []
    else:
        config.save_checkpoint_epochs = range(
            0, config.pt_num_epochs + 1, config.pt_ckpt_interval
        )

    if config.wandb:
        WANDB_PROJECT = f"simuclr_{config.dataset_name}_pt_ft"
        WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "SPECIFY_YOUR_WANDB_ENTITY")
        randint = random.randint(0, 1000)
        first_augs_str = "-".join(config.first_augs)
        if len(config.second_augs) == 0:
            second_augs_str = ""
            WANDB_NAME = f"stda_{first_augs_str}_{randint}"
        else:
            second_augs_str = "-".join(config.second_augs)
            WANDB_NAME = f"stda_{first_augs_str}_{second_augs_str}_{randint}"
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            mode="offline" if config.offline else "online",
            config=vars(config),
        )
        print(f"WandB initialized: {WANDB_PROJECT}/{WANDB_NAME}")
        setattr(config, "pt_run_name", WANDB_NAME)

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

    # Prepare Data Augmentation Planner
    first_augs = []
    second_augs = []
    for aug in config.first_augs:
        first_augs.append(prepare_aug_function(aug, config, device))

    if len(config.second_augs) != 0:
        for aug in config.second_augs:
            second_augs.append(prepare_aug_function(aug, config, device))
        aug_planner = ContrastiveAugPolicyPlanner(
            sub_policies=[first_augs, second_augs]
        )
    else:
        aug_planner = ContrastiveAugPolicyPlanner(sub_policies=[first_augs])

    # To store classification results
    frozen_train_results_list = []
    frozen_test_results_list = []

    unfrozen_train_results_list = []
    unfrozen_test_results_list = []

    start_time = time.time()
    for seed in config.seeds:
        print(paint(f"[*] Running with seed: {seed}"))
        # Init the weights again before nunning the code below.
        seed_torch(seed=seed)

        # aug_planner.current_subpolicies = [first_augs, second_augs]
        # Data loaders for pre-training
        train_loader = STDACLRDataLoader(
            dataset_train, aug_planner, config.pt_batch_size
        )
        val_loader = STDACLRDataLoader(dataset_val, aug_planner, config.pt_batch_size)

        if config.wandb:
            run_name = WANDB_NAME
        else:
            run_name = None

        # Pre-training
        simclr = pre_train_simclr(
            dataset_train,
            dataset_val,
            train_loader,
            val_loader,
            config,
            seed,
            run_name=run_name,
            device=device,
            save_checkpoint_epochs=config.save_checkpoint_epochs,
        )

        log_representation(simclr, dataset_test, seed, device, n_sample=50)

        ## Fine-tuning (using the pre-trained encoder)
        path_checkpoints = f"./logs/{config.pt_encoder}/{config.dataset_name}/pretrain_seed_{seed}/{config.pt_run_name}/checkpoints/"
        checkpoint_file = "model_best.pth"

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
                k=int(config.ft_data_fraction),
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
        for unfreeze in [False, True]:
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
                path_checkpoints=path_checkpoints,
                checkpoint_name=checkpoint_file,
                device=device,
                custom_sampler=custom_sampler,
                unfreeze=unfreeze,
            )

            loss_test, acc_test, fm_test, fw_test, elapsed = eval_model(
                model_ft, dataset_test, batch_size=256, seed=seed
            )

            # Save the results
            results_row = {
                "v_type": "split",
                "seed": seed,
                "checkpoint": checkpoint_file,
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
                "checkpoint": checkpoint_file,
                "test_loss": loss_test,
                "test_acc": acc_test,
                "test_fm": fm_test,
                "test_fw": fw_test,
            }

            if unfreeze:
                # Log the results to WandB
                wandb.log(
                    {
                        f"acc_test_{seed}_unfreeze": acc_test,
                        f"fm_test_{seed}_unfreeze": fm_test,
                        f"fw_test_{seed}_unfreeze": fw_test,
                    }
                )

                unfrozen_train_results_list.append(results_row)
                unfrozen_test_results_list.append(tests_results_row)
            else:  # Log the results to WandB
                wandb.log(
                    {
                        f"acc_test_{seed}": acc_test,
                        f"fm_test_{seed}": fm_test,
                        f"fw_test_{seed}": fw_test,
                    }
                )
                frozen_train_results_list.append(results_row)
                frozen_test_results_list.append(tests_results_row)

    # After the loop, convert lists of dictionaries to DataFrames

    unfrozen_train_results = pd.DataFrame(unfrozen_train_results_list)
    unfrozen_test_results = pd.DataFrame(unfrozen_test_results_list)

    frozen_train_results = pd.DataFrame(frozen_train_results_list)
    frozen_test_results = pd.DataFrame(frozen_test_results_list)

    wandb_logging(
        frozen_train_results,
        frozen_test_results,
        vars(config),
        epoch_name="ft_num_epochs",
        suffix="",
    )
    wandb_logging(
        unfrozen_train_results,
        unfrozen_test_results,
        vars(config),
        epoch_name="ft_num_epochs",
        suffix="_unfrozen",
    )
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
    print(paint("--" * 75, "blue"))
    wandb.finish()
