import sys
import glob

sys.path.append("..")
import time
import argparse
import random
import json
import pickle
import torch
import wandb
import numpy as np
import pandas as pd

from simuclr.datasets import SensorDataset, sliding_window
from simuclr.dataloaders import StratifiedSubsetSampler, FixedKPerClassSampler
from simuclr.augmentations.aug_planner import ContrastiveAugPolicyPlanner
from simuclr.dataloaders import PPDACLRDataLoader, RealWorldCustomSampler
from simuclr.utils import (
    get_default_device,
    seed_torch,
)
from simuclr.augmentations import ppda
from simuclr.utils import paint

from models.utils import eval_model, wandb_logging

from wimusim.datasets import WIMUSimDataset

from scripts.script_utils import pre_train_simclr, log_representation, fine_tuning
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
        "--ft_data_fraction",  # used for the stratified setting
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
        "--magscale_mean_center",
        action="store_true",
        help="Zero-center the scaling factor",
    )
    parser.add_argument(
        "--magwarp_sigma", type=float, default=0.2, help="Magnitude warping sigma"
    )
    parser.add_argument(
        "--magwarp_knot", type=int, default=2, help="Magnitude warping K knot"
    )
    parser.add_argument(
        "--magwarp_mean_center",
        action="store_true",
        help="Zero-center the scaling factor",
    )
    parser.add_argument(
        "--magscale_scale_translation", action="store_true", help="Scale translation"
    )
    parser.add_argument(
        "--magwarp_scale_translation", action="store_true", help="Scale translation"
    )
    parser.add_argument(
        "--timescale_min", type=float, default=0.8, help="Minimum time scaling factor"
    )
    parser.add_argument(
        "--timescale_max", type=float, default=1.2, help="Maximum time scaling factor"
    )
    # Actually, timewarp_sigma is not used in the code
    parser.add_argument(
        "--timewarp_sigma", type=float, default=0.2, help="Time warping sigma"
    )
    parser.add_argument(
        "--timewarp_knot", type=int, default=2, help="Time warping K knot"
    )
    parser.add_argument(
        "--timewarp_max_speed_ratio", type=float, default=2, help="Time warping K knot"
    )
    parser.add_argument(
        "--rotation_range_x",
        type=parse_range,
        default=(-25.0, 25.0),
        help="Rotation range for x-axis in the form 'min,max'",
    )
    parser.add_argument(
        "--rotation_range_y",
        type=parse_range,
        default=(-25.0, 25.0),
        help="Rotation range for y-axis in the form 'min,max'",
    )
    parser.add_argument(
        "--rotation_range_z",
        type=parse_range,
        default=(-25.0, 25.0),
        help="Rotation range for z-axis in the form 'min, max'",
    )
    parser.add_argument(
        "--rotation_paramix", action="store_true", help="Enable paramix for P params"
    )

    parser.add_argument(
        "--wimusim_params_path",
        type=str,
        default=f"../data/realworld/wimusim_params/",
        help="Path to WIMUSim parameters",
    )
    parser.add_argument(
        "--noisebias_sigma",
        type=float,
        default=0.1,
        help="Noise standard deviation (sigma)",
    )
    parser.add_argument(
        "--noisebias_bias_min",
        type=float,
        default=-1.0,
        help="Bias min value",
    )
    parser.add_argument(
        "--noisebias_bias_max",
        type=float,
        default=1.0,
        help="Bias max value",
    )

    # WandB logging
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--offline", action="store_true", help="Run wandb offline")

    return parser.parse_args()


def prepare_aug_function(aug_name: str, config, device):
    if aug_name == "magscale":
        mu = torch.tensor(config.magscale_mu, requires_grad=False)
        log_sigma = torch.tensor(
            config.magscale_sigma,
            requires_grad=False,
            # device=device,
        )
        magscale = ppda.MagnitudeScaling(
            mu,
            log_sigma,
            joint_names=config.magscale_joint_names,
            mean_center=config.magscale_mean_center,
            scale_translation=config.magscale_scale_translation,
        )
        return magscale
    elif aug_name == "magwarp":
        sigma = config.magwarp_sigma
        knot = config.magwarp_knot
        mean_center = config.magwarp_mean_center  # For MM-Fit use True

        magwarp = ppda.MagnitudeWarping(
            sigma=sigma,
            knot=knot,
            mean_center=mean_center,
            scale_translation=config.magwarp_scale_translation,
            device=device,
            joint_idx=config.magwarp_joint_idx,
        )
        return magwarp
    elif aug_name == "timescale":
        scale_min = config.timescale_min
        scale_max = config.timescale_max
        timescale = ppda.TimeScaling(
            scale_factor_min=scale_min,
            scale_factor_max=scale_max,  # device=device
        )
        return timescale
    elif aug_name == "timewarp":
        sigma = config.timewarp_sigma
        knot = config.timewarp_knot
        max_speed_ratio = config.timewarp_max_speed_ratio
        timewarp = ppda.TimeWarping(
            sigma=sigma,
            knot=knot,
            max_speed_ratio=max_speed_ratio,  # device=device
        )
        return timewarp
    elif aug_name == "rotation":
        rot_range_x = torch.tensor(config.rotation_range_x)  # , device=device)
        rot_range_y = torch.tensor(config.rotation_range_y)  # , device=device)
        rot_range_z = torch.tensor(config.rotation_range_z)  # , device=device)
        rotation = ppda.Rotation(
            rot_range_x,
            rot_range_y,
            rot_range_z,
            paramix=config.rotation_paramix,
            device=device,
        )
        return rotation
    elif aug_name == "noisebias":
        noise_bias = ppda.NoiseBias(
            noise_sigma=config.noisebias_sigma,
            bias_min=config.noisebias_bias_min,
            bias_max=config.noisebias_bias_max,
            device=device,
        )
        return noise_bias
    else:
        raise NotImplementedError("Invalid Data Augmentation name", aug_name)


def load_wimusim_params(dataset_name, split="train"):
    B_list, P_list, D_list, H_list = [], [], [], []
    groups = []  # Used to control the sampling process
    target_list = []

    if dataset_name == "realdisp":
        # Only load the first 10 subjects (will be modified when testing different splits)
        if split == "train":
            # Only load the first 10 subjects (will be modified when testing different splits)
            subject_ids = range(1, 11)
        elif split == "val":
            subject_ids = (11, 12)
        else:
            raise ValueError("Invalid split name")

        for subject_id in subject_ids:
            pkl_file_path = f"{config.wimusim_params_path}/realdisp_ideal_p{subject_id:03d}_wimusim_params_25Hz.pkl"
            with open(pkl_file_path, "rb") as f:
                wimusim_params_dict = pickle.load(f)
                B_list.append(wimusim_params_dict["B"])
                D_list.append(wimusim_params_dict["D"])
                P_list.append(wimusim_params_dict["P"])
                H_list.append(wimusim_params_dict["H"])
                target_list.append(wimusim_params_dict["target"])

    elif dataset_name == "realworld":
        if split == "train":
            # Only load the first 10 subjects (will be modified when testing different splits)
            subject_ids = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
        elif split == "val":
            subject_ids = [6, 7]
        else:
            raise ValueError("Invalid split name")

        for subject_id in subject_ids:
            sbj_pkl_files = glob.glob(
                f"{config.wimusim_params_path}/realworld_p{subject_id:03d}_*.pkl"
            )
            for pkl_file_path in sbj_pkl_files:
                print("Loading WIMUSim parameters from: ", pkl_file_path)
                with open(pkl_file_path, "rb") as f:
                    wimusim_params_dict = pickle.load(f)
                    B_list.append(wimusim_params_dict["B"])
                    D_list.append(wimusim_params_dict["D"])
                    P_list.append(wimusim_params_dict["P"])
                    H_list.append(wimusim_params_dict["H"])
                    target_list.append(wimusim_params_dict["target"])
    elif dataset_name == "mmfit":
        if split == "train":
            # Only load the first 10 subjects (will be modified when testing different splits)
            recording_ids = [0, 3, 7, 12, 13, 16]
        elif split == "val":
            recording_ids = [17, 19]
        else:
            raise ValueError("Invalid split name")

        for recording_id in recording_ids:
            sbj_pkl_files = glob.glob(
                f"{config.wimusim_params_path}/w{recording_id:02d}_wimusim_params_dict.pkl"
            )
            for pkl_file_path in sbj_pkl_files:
                print("Loading WIMUSim parameters from: ", pkl_file_path)
                with open(pkl_file_path, "rb") as f:
                    wimusim_params_dict = pickle.load(f)
                    B_list.append(wimusim_params_dict["B"])
                    D_list.append(wimusim_params_dict["D"])
                    P_list.append(wimusim_params_dict["P"])
                    H_list.append(wimusim_params_dict["H"])
                    target_list.append(wimusim_params_dict["target"])

    print(
        "Moving all the parameters to CPU (to use multiple workers in the dataloader)"
    )
    for B in B_list:
        for key, b in B.rp.items():
            B.rp[key] = b.detach().cpu()

    for D in D_list:
        for key, d in D.translation.items():
            D.translation[key] = d.detach().cpu()
        for key, d in D.orientation.items():
            D.orientation[key] = d.detach().cpu()
    for P in P_list:
        for key, p in P.rp.items():
            P.rp[key] = p.detach().cpu()
        for key, p in P.ro.items():
            P.ro[key] = p.detach().cpu()
    for H in H_list:
        for key, h in H.ba.items():
            H.ba[key] = h.detach().cpu()
        for key, h in H.sa.items():
            H.sa[key] = h.detach().cpu()
        for key, h in H.bg.items():
            H.bg[key] = h.detach().cpu()
        for key, h in H.sg.items():
            H.sg[key] = h.detach().cpu()

    return B_list, P_list, D_list, H_list, target_list


def prepare_dataloaders(
    config, wimusim_dataset_train, wimusim_dataset_val, aug_planner, seed
):
    if config.dataset_name == "realworld":
        # Data loaders for pre-training
        sampler_train = RealWorldCustomSampler(
            dataset=wimusim_dataset_train,
            batch_size=config.pt_batch_size,
            max_list_ids_per_batch=4,  # 2 ** 2
        )
        sampler_val = RealWorldCustomSampler(
            dataset=wimusim_dataset_val,
            batch_size=config.pt_batch_size,
            max_list_ids_per_batch=4,  # 2 ** 2
        )

        train_sim_loader = PPDACLRDataLoader(
            wimusim_dataset_train,
            batch_size=config.pt_batch_size,
            worker_init_fn=np.random.seed(int(seed)),
            aug_planner=aug_planner,
            sampler=sampler_train,  # For RealWorldDataset
        )
        val_sim_loader = PPDACLRDataLoader(
            wimusim_dataset_val,
            batch_size=config.pt_batch_size,
            worker_init_fn=np.random.seed(int(seed)),
            aug_planner=aug_planner,
            sampler=sampler_val,  # For RealWorldDataset
        )
    elif config.dataset_name in ["realdisp", "mmfit"]:
        train_sim_loader = PPDACLRDataLoader(
            wimusim_dataset_train,
            batch_size=config.pt_batch_size,
            shuffle=True,
            worker_init_fn=np.random.seed(int(seed)),
            aug_planner=aug_planner,
        )
        val_sim_loader = PPDACLRDataLoader(
            wimusim_dataset_val,
            batch_size=config.pt_batch_size,
            shuffle=True,
            worker_init_fn=np.random.seed(int(seed)),
            aug_planner=aug_planner,
        )

    else:
        raise NotImplementedError("Invalid dataset name")

    return train_sim_loader, val_sim_loader


if __name__ == "__main__":
    config = parse_args()
    configure_dataset_params(config)

    # Add path_processed to the config
    setattr(config, "path_processed", f"../data/{config.dataset_name}/")
    setattr(
        config, "wimusim_params_path", f"../data/{config.dataset_name}/wimusim_params/"
    )

    if config.dataset_name in ["realdisp", "realworld"]:
        config.magscale_joint_names = [
            "PELVIS",
            "R_SHOULDER",
            "R_ELBOW",
            "R_HIP",
            "R_KNEE",
            "L_SHOULDER",
            "L_ELBOW",
            "L_HIP",
            "L_KNEE",
        ]
        config.magwarp_joint_idx = None
    elif config.dataset_name == "mmfit":
        config.magscale_joint_names = [
            "NECK",
            "R_SHOULDER",
            "L_SHOULDER",
        ]
        config.magwarp_joint_idx = [10, 15, 18]
    else:
        raise NotImplementedError("Invalid dataset name")

    if config.pt_ckpt_interval == -1:
        config.save_checkpoint_epochs = []
    else:
        config.save_checkpoint_epochs = range(
            0, config.pt_num_epochs + 1, config.pt_ckpt_interval
        )

    if config.wandb:
        WANDB_PROJECT = f"simuclr_{config.dataset_name}_pt_ft"
        # WANDB_ENTITY = "nobuyuki"
        randint = random.randint(0, 1000)
        first_augs_str = "-".join(config.first_augs)
        if len(config.second_augs) == 0:
            second_augs_str = ""
            WANDB_NAME = f"ppda_{first_augs_str}_{randint}"
        else:
            second_augs_str = "-".join(config.second_augs)
            WANDB_NAME = f"ppda_{first_augs_str}_{second_augs_str}_{randint}"
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

    B_list, P_list, D_list, H_list, target_list = load_wimusim_params(
        config.dataset_name, split="train"
    )

    wimusim_dataset_train = WIMUSimDataset(
        B_list=B_list,
        D_list=D_list,
        P_list=P_list,
        H_list=H_list,
        target_list=target_list,
        window=config.window,
        stride=config.stride,
        acc_only=False,
        scale_config=scaling_config,
    )

    B_list, P_list, D_list, H_list, target_list = load_wimusim_params(
        config.dataset_name, split="val"
    )

    wimusim_dataset_val = WIMUSimDataset(
        B_list=B_list,
        D_list=D_list,
        P_list=P_list,
        H_list=H_list,
        target_list=target_list,
        window=config.window,
        stride=config.stride,
        acc_only=False,
        scale_config=scaling_config,
    )

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
        train_sim_loader, val_sim_loader = prepare_dataloaders(
            config, wimusim_dataset_train, wimusim_dataset_val, aug_planner, seed
        )

        print(paint(f"[*] Running with seed: {seed}"))
        # Init the weights again before nunning the code below.
        seed_torch(seed=seed)

        if config.wandb:
            run_name = WANDB_NAME
        else:
            run_name = None

        # Pre-training
        simclr = pre_train_simclr(
            dataset_train,
            dataset_val,
            train_sim_loader,
            val_sim_loader,
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
            # pick k samples per class
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
