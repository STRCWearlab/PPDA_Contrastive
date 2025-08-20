import sys
import glob
from collections import Counter

sys.path.append("..")
import time
import argparse
import random
import json
import wandb
import pandas as pd

from simuclr.datasets import SensorDataset, sliding_window
from simuclr.dataloaders import StratifiedSubsetSampler, FixedKPerClassSampler
from simuclr.utils import (
    get_default_device,
    seed_torch,
)
from simuclr.utils import paint

from models.utils import eval_model, wandb_logging

from scripts.script_utils import fine_tuning
from scripts.simuclr_pt_ft_stda import configure_dataset_params


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
        "--dataset_name", type=str, default="realworld", help="Name of dataset"
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
    parser.add_argument(
        "--pt_run_name", type=str, default="wandb-runname", help="WandB run name"
    )
    parser.add_argument("--pt_encoder", type=str, default="TPN", help="Encoder model")
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
        "--ft_data_sample_type", type=str, default="k_sample", help="Data sampling type"
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

    # WandB logging
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--offline", action="store_true", help="Run wandb offline")

    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    configure_dataset_params(config)

    # Add path_processed to the config
    setattr(config, "path_processed", f"../data/{config.dataset_name}/")

    if config.wandb:
        WANDB_PROJECT = f"simuclr_{config.dataset_name}_ft_ld"
        # WANDB_ENTITY = "nobuyuki"
        randint = random.randint(0, 1000)

        WANDB_NAME = f"{config.pt_run_name}_ft_{randint}"
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            mode="offline" if config.offline else "online",
            config=vars(config),
        )
        print(f"WandB initialized: {WANDB_PROJECT}/{WANDB_NAME}")
        # setattr(config, "pt_run_name", WANDB_NAME)
        setattr(config, "run_name", WANDB_NAME)

    # Get the default device (use cuda if available)
    device = get_default_device()
    print(paint(f"[*] Using device: {device}"))

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

        if config.wandb:
            run_name = WANDB_NAME
        else:
            run_name = None

        ## Fine-tuning (using the pre-trained encoder)
        path_checkpoints = f"./logs/{config.pt_encoder}/{config.dataset_name}/pretrain_seed_{seed}/{config.pt_run_name}/checkpoints/"
        checkpoint_file = "model_best.pth"

        _, train_target_list = sliding_window(
            dataset_train.data,
            dataset_train.target,
            dataset_train.window,
            dataset_train.stride,
        )

        ### Comment out if you want to see how many samples each class has
        # label_counts = Counter(train_target_list)
        # for label, count in sorted(label_counts.items()):
        #     print(f"Class {label}: {count} samples in total")

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
        for unfreeze in [False, True]:
            print(path_checkpoints)
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
                run_name=WANDB_NAME,
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
