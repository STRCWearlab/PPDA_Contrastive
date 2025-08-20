import copy
import random
import time

import torch
from itertools import chain
from torch.utils.data import Sampler
from collections import defaultdict
from torch.utils.data import DataLoader
import wimusim
from wimusim.datasets import WIMUSimDataset

from simuclr.datasets import SensorDataset
from simuclr.augmentations import stda, ppda
from sklearn.model_selection import train_test_split
import numpy as np


## For n_sub_baseline evaluations
class FixedLengthDataLoader(DataLoader):
    def __init__(self, dataset, fixed_length, *args, **kwargs):
        super(FixedLengthDataLoader, self).__init__(dataset, *args, **kwargs)
        self.fixed_length = fixed_length

    def __len__(self):
        return self.fixed_length  # Use specified number of batches

    def __iter__(self):
        base_iter = super(FixedLengthDataLoader, self).__iter__()
        for _ in range(self.fixed_length):
            try:
                X, y, idx = next(base_iter)
            except StopIteration:
                # Restart the base iterator when data is exhausted
                base_iter = super(FixedLengthDataLoader, self).__iter__()
                X, y, idx = next(base_iter)
            yield X, y, idx


class STDACLRDataLoader(DataLoader):
    def __init__(self, dataset, aug_planner, *args, **kwargs):
        """
        Custom DataLoader for SimCLR.
        :param dataset: Input dataset.
        :param aug_planner: Instance of ContrastiveAugPolicyPlanner.
        """
        super(STDACLRDataLoader, self).__init__(dataset, *args, **kwargs)
        self.aug_planner = aug_planner

    def __iter__(self):
        for batch in super(STDACLRDataLoader, self).__iter__():
            X, y, idx = batch  # Assumes dataset returns (data, labels)

            # Sample two augmentation policies
            policy1, policy2 = self.aug_planner.sample_policies()

            # Generate two views
            view1 = self.apply_augmentations(X, idx, policy1)
            view2 = self.apply_augmentations(X, idx, policy2)

            yield view1, view2, idx

    def apply_augmentations(self, X, idx, policy):
        """
        Apply a sequence of augmentations (policy) to the input data.
        Handles augmentations requiring special logic.
        :param X: Input data (batch of time-series).
        :param idx: Indices of the samples in the dataset.
        :param policy: Sequence of augmentation operations.
        :return: Augmented data.
        """
        for augmentor in policy:
            if isinstance(augmentor, stda.TimeScaling):
                # Sample a scaling factor
                scale_factor = augmentor.sample_scaling_factor()
                T_orig = X.shape[1]  # Time dimension (N, T, C)

                if scale_factor > 1.0:
                    # Re-fetch data with the scaled window size
                    X = torch.stack(
                        [
                            self.dataset.__getitem__(i, scale=scale_factor)[0]
                            for i in idx
                        ]
                    )

                # Apply time scaling
                X = augmentor.apply_augmentation(X=X, T_orig=T_orig)

            else:
                # Default augmentation application
                X = augmentor.apply_augmentation(X=X)

        return X


class PPDACLRDataLoader(DataLoader):
    dataset: WIMUSimDataset

    def __init__(
        self,
        dataset,
        aug_planner,
        n_batches_per_epoch=None,
        num_workers=8,
        *args,
        **kwargs,
    ):
        """
        Custom DataLoader for SimCLR.
        :param dataset: Input dataset.
        :param aug_planner: Instance of ContrastiveAugPolicyPlanner.
        :param device: Device to use.
        :param n_batches_per_epoch: Number of batches per epoch.
        """
        super(PPDACLRDataLoader, self).__init__(
            dataset, num_workers=num_workers, pin_memory=True, *args, **kwargs
        )
        self.aug_planner = aug_planner
        self.n_batches_per_epoch = n_batches_per_epoch
        self.device = dataset.device
        self._init_wimusim_env()

    def __iter__(self):
        # Better to avoid using GPU for the iteration process to utilize multiple CPU cores
        base_iter = super(PPDACLRDataLoader, self).__iter__()
        batch_count = 0

        while True:
            try:
                # Get the next batch from the base iterator
                X_d, y, (idx, list_ids) = next(base_iter)
            except StopIteration:
                if self.n_batches_per_epoch is None:
                    break
                # Restart the base iterator when data is exhausted
                base_iter = super(PPDACLRDataLoader, self).__iter__()
                X_d, y, (idx, list_ids) = next(base_iter)

            # Sample two augmentation policies
            policy1, policy2 = self.aug_planner.sample_policies()

            # Generate two views
            view1 = self.apply_augmentations(X_d, y, idx, list_ids, policy1)
            view2 = self.apply_augmentations(X_d, y, idx, list_ids, policy2)

            # Scale the data (virtual IMU)
            view1_scaled = self.scale_data(view1)
            view2_scaled = self.scale_data(view2)
            batch_count += 1

            if (
                self.n_batches_per_epoch is not None
                and batch_count >= self.n_batches_per_epoch
            ):
                break

            yield view1_scaled, view2_scaled, idx

    def __len__(self):
        if self.n_batches_per_epoch is not None:
            return self.n_batches_per_epoch  # Use specified number of batches
        else:
            return super().__len__()

    def _init_wimusim_env(self):
        self.wimusim_env = wimusim.WIMUSim(
            B=self.dataset.B_list[0],
            D=self.dataset.D_list[0],
            P=self.dataset.P_list[0],
            H=self.dataset.H_list[0],
            device="cpu",  # Use CPU for simulation
        )

    def scale_data(self, X):
        """
        Scale the input data.
        :param X: Input data (batch of time-series).
        :param scale_factor: Scaling factor.
        :return: Scaled data.
        """
        X_scaled = (
            (
                (X.detach().cpu() - self.dataset.scale_config["mean"])
                / self.dataset.scale_config["std"]
            )
            # .to(self.device)
            .float()
        )

        return X_scaled

    def apply_augmentations(self, X_d, y, idx, list_ids, policy):
        """
        Apply a sequence of augmentations (policy) to the input data.
        Handles augmentations requiring special logic.
        :param X: Input data (batch of time-series).
        :param idx: Indices of the samples in the dataset.
        :param list_idx: Indices of the subjects in the dataset.
        :param policy: Sequence of augmentation operations.
        :return: Augmented data.
        """
        virtual_IMU_dict = {
            imu_name: {"acc": [], "gyro": []}
            for imu_name in self.dataset.P_list[0].imu_names
        }

        for augmentor in policy:
            # TimeScaling should be applied to the data.
            if isinstance(augmentor, ppda.TimeScaling):
                # Sample a scaling factor and re-fetch data with the new scale
                scale_factor: float = augmentor.sample_scaling_factor()
                T_orig = X_d.shape[2]  # T is the time dimension (N, J, T, C)
                if scale_factor > 1.0:
                    # Add this line to prevent index out of bounds (possibly due to some error in the CustomSampler)
                    idx = torch.clamp(idx, 0, X_d.shape[0] - 1)
                    # Re-fetch data with the scaled window size
                    start_time = time.time()
                    X_d = torch.stack(
                        [
                            self.dataset.__getitem__(i, scale=scale_factor)[0]
                            for i in idx
                        ]
                    )
                    end_time = time.time()
                    # print(
                    #     f"Time taken to re-fetch data: {end_time - start_time:.4f} seconds"
                    # )
                    # print("X_d", X_d.device)

                    # Loading the entire dataset is faster than loading one by one with randomized indices
                    # Was it faster? Now the above implementation is actually faster
                    # start_time = time.time()
                    # X_d = torch.stack(
                    #     [
                    #         self.dataset.__getitem__(i, scale=scale_factor)[0]
                    #         for i in range(self.dataset.__len__())
                    #     ]
                    # )[idx]
                    # end_time = time.time()
                    # print(
                    #     f"Time taken to re-fetch data: {end_time - start_time:.4f} seconds"
                    # )
                # Apply time scaling if needed
                X_d = augmentor.apply_augmentation(X=X_d, T_orig=T_orig)
            elif isinstance(augmentor, ppda.TimeWarping):
                X_d = augmentor.apply_augmentation(X=X_d)
            elif isinstance(augmentor, ppda.MagnitudeWarping):
                X_d = augmentor.apply_augmentation(X=X_d)

        # Data Augmentation involving B, P, H are applied here
        for list_id in torch.unique(list_ids):
            # print(f"processing list_id {list_id}...")
            X_d_sbj = X_d[list_ids == list_id]  # Get the data of the subject.
            # y_sbj = y[list_ids == list_id]  # Get the target of the subject.
            # idx_sbj = idx[list_ids == list_id]  # Get the index of the subject.

            B = copy.deepcopy(self.dataset.B_list[list_id])
            P = copy.deepcopy(self.dataset.P_list[list_id])
            H = copy.deepcopy(
                self.dataset.H_list[list_id]
            )  # TODO: We don't really need to use this as it will be overwritten by the WIMUSim object
            # print(B, P, H)
            # We can use the same TimeScaling
            for augmentor in policy:
                if isinstance(augmentor, ppda.Rotation):
                    if augmentor.paramix:
                        # Select P randomly from the P_list if paramix is enabled
                        P = copy.deepcopy(random.choice(self.dataset.P_list))
                    P = augmentor.apply_augmentation(P=P)
                elif isinstance(augmentor, ppda.NoiseBias):
                    if hasattr(H, "imu_names"):
                        H = augmentor.apply_augmentation(H=H)
                    else:
                        # Use P.imu_names if H does not have imu_names
                        H = augmentor.apply_augmentation(H=H, imu_names=P.imu_names)

            # Transform D_data to WIMUSim.Dynamics
            # It's okay to apply modification directly to the D as its newly created.
            # warnings.warn(
            #     "Check the sampling rate of the dataset. Currently, sample_rate=30 is hardcoded."
            # )
            if type(X_d_sbj) == torch.Tensor:
                X_d_sbj = X_d_sbj.detach().cpu().numpy()

            D = wimusim.WIMUSim.Dynamics(
                translation={"XYZ": X_d_sbj[:, 0, :, 1:]},
                orientation={
                    joint_name: X_d_sbj[:, joint_id, :, :]
                    for joint_name, joint_id in self.dataset._D_ori_key_idx.items()
                },
                sample_rate=self.wimusim_env.D.sample_rate,
                # device=self.device,
            )

            # Apply the augmentations from the current subpolicy
            # for augmentor in self.aug_planner.current_subpolicy:
            for augmentor in policy:
                if isinstance(augmentor, ppda.MagnitudeScaling):
                    D = augmentor.apply_augmentation(D=D)

            if self.wimusim_env is None:
                self.wimusim_env = wimusim.WIMUSim(B, D, P, H, device=self.device)
            else:
                # Reuse the WIMUSim object if it already exists
                self.wimusim_env.B = B
                self.wimusim_env.D = D
                self.wimusim_env.P = P
                self.wimusim_env.H = H

            # self.wimusim_env.move_onto_device()  # Move to GPU # Do everything on CPU
            virtual_IMU_dict_sbj = self.wimusim_env.simulate(
                mode="generate", vectorize=True
            )
            for imu_name, imu_data in virtual_IMU_dict_sbj.items():
                virtual_IMU_dict[imu_name]["acc"].append(imu_data[0])
                virtual_IMU_dict[imu_name]["gyro"].append(imu_data[1])

            if self.dataset.acc_only:
                X_res = torch.concat(
                    [
                        torch.concat(virtual_IMU_dict[imu_name]["acc"], dim=0)
                        for imu_name in self.dataset.P_list[0].imu_names
                    ],
                    dim=-1,
                )
            elif self.dataset.gyro_only:
                X_res = torch.concat(
                    [
                        torch.concat(virtual_IMU_dict[imu_name]["gyro"], dim=0)
                        for imu_name in self.dataset.P_list[0].imu_names
                    ],
                    dim=-1,
                )
            else:
                if self.dataset.data_order == "alternate":
                    X_res = torch.concat(
                        [
                            torch.concat(virtual_IMU_dict[imu_name][sensor_type], dim=0)
                            for imu_name in self.dataset.P_list[0].imu_names
                            for sensor_type in ["acc", "gyro"]
                        ],
                        dim=-1,
                    )
                elif self.dataset.data_order == "sequential":
                    X_res = torch.concat(
                        [
                            torch.concat(virtual_IMU_dict[imu_name][sensor_type], dim=0)
                            for sensor_type in ["acc", "gyro"]
                            for imu_name in self.dataset.P_list[0].imu_names
                        ],
                        dim=-1,
                    )
                else:
                    raise ValueError("Invalid order", self.dataset.data_order)
        return X_res


class RealWorldCustomSampler(Sampler):
    def __init__(self, dataset, batch_size, max_list_ids_per_batch=13):
        """
        :param dataset: The dataset to sample from
        :param batch_size: Number of samples in each batch
        :param max_list_ids_per_batch: Maximum number of unique list_ids in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_list_ids_per_batch = max_list_ids_per_batch

        # Use a dictionary to map class labels to list_ids
        self.class_to_list_ids = defaultdict(list)
        for list_id, t_list in enumerate(dataset.target_list):
            target = torch.unique(t_list)[0].item()
            self.class_to_list_ids[target].append(list_id)

    def __iter__(self):
        # Step 0: Prepare the original and working lists of indices
        list_id_to_indices_ref = {
            list_id: list(index_range)
            for list_id, index_range in self.dataset._index_range_dict.items()
        }
        class_to_list_ids = copy.deepcopy(self.class_to_list_ids)

        # Debug: Print at the beginning of each epoch
        # print("\n--- New Epoch ---")

        # Step 1: Group samples by list_id based on `_index_range_dict`
        # Each list_id is mapped to a list of sample indices in its range
        list_id_to_indices = copy.deepcopy(list_id_to_indices_ref)

        # Step 2: Shuffle indices within each list_id group to ensure randomness
        for indices in list_id_to_indices.values():
            random.shuffle(indices)

        # Step 3: Generate the sequence of indices while respecting the max number of unique list_ids per batch
        batches = []
        available_classes = list(class_to_list_ids.keys())

        # Debug: Confirm reset state
        # print("Available classes at epoch start:", available_classes)
        # print("Initial length of list_id_to_indices:")
        # for list_id, indices in list_id_to_indices.items():
        #     print(f"List ID {list_id}: {len(indices)} indices")

        while available_classes:
            # print(f"Available classes: {available_classes}")
            # Randomly select up to `max_list_ids_per_batch` unique list_ids
            selected_classes = random.sample(
                available_classes,
                min(self.max_list_ids_per_batch, len(available_classes)),
            )

            selected_list_ids = []
            for activity_class in selected_classes:
                available_list_ids = [
                    list_id
                    for list_id in class_to_list_ids[activity_class]
                    if list_id_to_indices[list_id]
                ]
                if available_list_ids:
                    selected_list_ids.append(random.choice(available_list_ids))
            # Debug: Check selected classes and list IDs
            # print("Selected classes:", selected_classes)
            # print("Selected list_ids:", selected_list_ids)

            if len(selected_list_ids) == 0:
                break

            # print(f"Selected list_ids: {selected_list_ids}")
            # Collect indices for the selected list_ids to form a batch
            total_remaining_samples = sum(
                len(list_id_to_indices[list_id]) for list_id in selected_list_ids
            )
            batch_indices = []
            for list_id in selected_list_ids:
                remaining_samples = len(list_id_to_indices[list_id])
                take_count = max(
                    1,
                    int(
                        self.batch_size * (remaining_samples / total_remaining_samples)
                    ),
                )
                # Ensure we don’t exceed available samples
                take_count = min(take_count, len(list_id_to_indices[list_id]))
                batch_indices.extend(list_id_to_indices[list_id][:take_count])
                list_id_to_indices[list_id] = list_id_to_indices[list_id][take_count:]

                # Remove list_id from available_list_ids if no more samples remain
                if not list_id_to_indices[list_id]:
                    # Loop through each activity class and find the class that contains the depleted list_id
                    for activity_class, list_ids in class_to_list_ids.items():
                        if list_id in list_ids:
                            # Remove the depleted list_id from its class's list_ids
                            list_ids.remove(list_id)
                            depleted_class = activity_class
                            break
                    # print(f"List_id {list_id} is depleted of class {depleted_class}")
                    # print(
                    #     f"remaining no of {depleted_class}: ",
                    #     len(self.class_to_list_ids[depleted_class]),
                    # )
                    # If the activity class no longer has any list_ids with samples, remove it from available_classes
                    if depleted_class and not class_to_list_ids[depleted_class]:
                        available_classes.remove(depleted_class)

            # Adjust batch size if necessary by trimming or duplicating selected samples
            if len(batch_indices) < self.batch_size:
                # If we have not reached the batch size, we need to add more list_ids
                if len(selected_list_ids) < self.max_list_ids_per_batch:
                    # Add more list_ids to the batch
                    additional_list_ids = random.sample(
                        list(list_id_to_indices_ref.keys()),
                        self.max_list_ids_per_batch - len(selected_list_ids),
                    )
                    selected_list_ids.extend(additional_list_ids)

                additional_indices = list(
                    chain.from_iterable(
                        list_id_to_indices_ref[list_id] for list_id in selected_list_ids
                    )
                )
                random.shuffle(additional_indices)
                batch_indices.extend(
                    additional_indices[: self.batch_size - len(batch_indices)]
                )

            # Shuffle and add to batches
            random.shuffle(batch_indices)
            batches.append(batch_indices)

        # Step 4: Shuffle the batches to add randomness across the epoch and return the iterator
        # random.shuffle(batches)
        return iter(chain.from_iterable(batches))

    def __len__(self):
        return sum(len(indices) for indices in self.dataset._index_range_dict.values())


class StratifiedSubsetSampler(Sampler):
    """
    Custom Sampler for selecting a stratified subset of the dataset.

    Parameters:
    - labels: The entire list or array of dataset labels.
    - fraction: Fraction of data to use (e.g., 0.1 for 10% labeled data).
    - shuffle: Whether to shuffle the data before sampling.
    - seed: Random seed for reproducibility.
    """

    def __init__(self, labels, fraction=1.0, shuffle=True, seed=42):
        super().__init__(data_source=None)
        self.labels = np.array(labels)
        self.fraction = fraction
        self.seed = seed
        self.shuffle = shuffle
        if self.fraction < 1.0:
            self.indices = self._get_subset_indices()
        else:
            self.indices = np.arange(len(self.labels))

    def _get_subset_indices(self):
        """Perform stratified sampling to select indices for the given fraction."""
        num_samples = len(self.labels)
        num_selected = max(
            1, int(num_samples * self.fraction)
        )  # Ensure at least 1 sample

        # Stratified split
        indices = np.arange(num_samples)
        selected_indices, _ = train_test_split(
            indices,
            train_size=num_selected,
            stratify=self.labels,
            random_state=self.seed,
        )

        return selected_indices

    def __iter__(self):
        """Returns an iterator over the selected indices."""
        indices = self.indices.copy()  # Copy to avoid modifying original order
        if self.shuffle:
            np.random.shuffle(indices)  # Explicitly shuffle
        return iter(indices)

    def __len__(self):
        """Returns the number of selected samples."""
        return len(self.indices)


class FixedKPerClassSampler(Sampler):
    """
    Custom Sampler that selects up to k samples per class.

    Parameters:
    - labels: A list or array of dataset labels.
    - k: Max number of samples to select per class.
    - seed: Random seed for reproducibility.
    - shuffle: Whether to shuffle the selected samples.
    """

    def __init__(self, labels, k, seed=42, shuffle=True):
        super().__init__(data_source=None)
        self.labels = np.array(labels)
        self.k = k
        self.seed = seed
        self.shuffle = shuffle
        self.indices = self._get_k_per_class_indices()

    def _get_k_per_class_indices(self):
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label_to_indices[label].append(idx)

        random.seed(self.seed)
        selected_indices = []
        for label, indices in label_to_indices.items():
            if len(indices) <= self.k:
                selected = indices
            else:
                selected = random.sample(indices, self.k)
            selected_indices.extend(selected)

        if self.shuffle:
            random.shuffle(selected_indices)

        return selected_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
