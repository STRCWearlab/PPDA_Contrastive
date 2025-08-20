import warnings
from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from pytorch3d.transforms import euler_angles_to_matrix

from simuclr.utils import get_default_device, paint
from simuclr.augmentations import (
    BaseRotation,
    BaseMagnitudeScaling,
    BaseTimeScaling,
    BaseMagnitudeWarping,
    BaseTimeWarping,
    BaseJittering,
)


class Jittering(BaseJittering):
    def apply_augmentation(self, X: torch.Tensor, columns=None):
        """
        Apply scaling to specified columns of a (N, T, C) dataset.

        Parameters:
        - X: Input data of shape (N, T, C), where N is the batch size, T is window length and C is the number of channels.
        - sigma: Standard deviation of the Gaussian distribution used for generating the scaling factor.
        - columns: A list of column indices to apply the scaling. If None, applies to all columns.

        Returns:
        - Scaled data with the same shape as the input.
        """

        if columns is not None:
            raise NotImplementedError(
                "Scaling only implemented for all columns for now"
            )

        noise = self.sample_noise(X.shape)
        X_aug = X + noise  # Broadcast scaling factors to all elements in the batch
        return X_aug


class MagnitudeScaling(BaseMagnitudeScaling):
    def apply_augmentation(self, X: torch.Tensor, columns=None):
        """
        Apply scaling to specified columns of a (N, T, C) dataset.

        Parameters:
        - X: Input data of shape (N, T, C), where N is the batch size, T is window length and C is the number of channels.
        - sigma: Standard deviation of the Gaussian distribution used for generating the scaling factor.
        - columns: A list of column indices to apply the scaling. If None, applies to all columns.

        Returns:
        - Scaled data with the same shape as the input.
        """

        if columns is not None:
            raise NotImplementedError(
                "Scaling only implemented for all columns for now"
            )
        batch_size = X.shape[0]
        scaling_factors = self.sample_scaling_factors(batch_size, cpu=True)
        # print(X.device, scaling_factors.device)
        X_aug = X * scaling_factors.view(
            batch_size, 1, 1
        )  # Broadcast scaling factors to all elements in the batch
        return X_aug


class TimeWarping(BaseTimeWarping):
    def apply_augmentation(self, X: torch.Tensor) -> torch.Tensor:
        """
            Apply time warping to a (N, T, C) dataset, leveraging PchipInterpolator with axis=1,
            to interpolate along the time dimension for the entire batch at once.

            Parameters:
            - x: Input data of shape (N, T, C).
            - n_speed_change: The number of speed changes in the time warp.
            - max_speed_ratio: The maximum ratio of speed change in the warp.
            - seed: Optional seed for random number generator.

            Returns:
            - Warp

        # def time_warp(x, n_speed_change=2, max_speed_ratio=1.5, seed=None):

           ed data with the same shape as the input.
        """

        X_type = type(X)
        if X_type == torch.Tensor:
            X = X.detach().cpu().numpy()

        T = X.shape[1]  # (N, T, C)

        # Generate common anchor points and warped indices for the entire batch
        anchors = np.linspace(0, T - 1, num=self.knot + 2)
        if isinstance(self.max_speed_ratio, (list, tuple)):
            max_speed_ratio_value = np.random.uniform(
                low=self.max_speed_ratio[0], high=self.max_speed_ratio[1]
            )
        else:
            max_speed_ratio_value = self.max_speed_ratio

        segment_speeds = np.random.uniform(
            1.0, max_speed_ratio_value, size=(self.knot + 1,)
        )
        segment_speeds /= (
            segment_speeds.mean()
        )  # Normalize to maintain overall duration
        new_anchors = np.concatenate(
            (np.array([0]), np.cumsum(np.diff(anchors) * segment_speeds))
        )

        # Generate the warped time indices using PchipInterpolator
        warp_interpolator = PchipInterpolator(
            new_anchors, anchors, axis=0, extrapolate=True
        )
        warped_indices = warp_interpolator(np.arange(T))

        # Interpolate the original data onto the new, warped timeline for the entire batch
        data_interpolator = PchipInterpolator(np.arange(T), X, axis=1, extrapolate=True)
        X_aug = data_interpolator(warped_indices)

        # Make it a torch tensor
        return torch.tensor(X_aug, dtype=torch.float32)


class MagnitudeWarping(BaseMagnitudeWarping):
    def apply_augmentation(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping to a (B, T, N) dataset using PchipInterpolator.

        Parameters:
        - X: Input data of shape (N, T, C), where N is batch size, T is time length, and C is the number of channels.

        Returns:
        - Magnitude warped data with the same shape as the input.
        """
        # Convert to numpy if it's a torch tensor
        X_type = type(X)
        if X_type == torch.Tensor:
            X = X.detach().cpu().numpy()

        B, T, N = X.shape  # (Batch size, Time length, Number of channels)

        # Generate common anchor points and magnitude warps for the entire batch
        anchors = np.linspace(0, T - 1, num=self.knot + 2)
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(self.knot + 2,)
        )

        # Generate the magnitude warp indices using PchipInterpolator
        spline_interpolator = CubicSpline(anchors, random_warps, axis=0)
        magnitude_warps = spline_interpolator(np.arange(T))

        # Apply magnitude warping for the entire batch by multiplying the warping factors
        X_aug = X * magnitude_warps[np.newaxis, :, np.newaxis]

        # Convert back to torch tensor
        return torch.tensor(X_aug, dtype=torch.float32)


class Rotation(BaseRotation):
    def apply_augmentation(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the learnable rotation to 3D data points.

        Parameters:
        - x: Input data of shape (N, T, C).

        Returns:
        - Rotated data with the same shape as the input.
        """
        # Get the learnable rotation angles for each axis
        (
            min_angle_x,
            max_angle_x,
            min_angle_y,
            max_angle_y,
            min_angle_z,
            max_angle_z,
        ) = self.get_rotation_angles()

        # Generate random angles within the learnable range for each axis
        angle_x = torch.rand(1).item() * (max_angle_x - min_angle_x) + min_angle_x
        angle_y = torch.rand(1).item() * (max_angle_y - min_angle_y) + min_angle_y
        angle_z = torch.rand(1).item() * (max_angle_z - min_angle_z) + min_angle_z

        # Convert angles to radians
        angles_rad = torch.deg2rad(torch.tensor([angle_x, angle_y, angle_z]))

        # Convert Euler angles to a rotation matrix (3x3)
        rotation_matrix = euler_angles_to_matrix(angles_rad, convention="XYZ")

        # Reshape input data: (N, T, C) -> (N*T, 3)
        N, T, C = X.shape
        assert (
            C % 3 == 0
        ), "C must be divisible by 3 (each set of 3 channels represents a 3D point)"
        reshaped_x = X.reshape(-1, 3)

        # Apply the rotation
        rotated_x = torch.matmul(reshaped_x, rotation_matrix.T)

        # Reshape back to original shape (N, T, C)
        rotated_x = rotated_x.view(N, T, C)

        return rotated_x


class TimeScaling(BaseTimeScaling):
    def apply_augmentation(self, X: torch.Tensor, T_orig: int = None) -> torch.Tensor:
        """
        Apply time scaling to a (N, T, C) dataset, where N is the batch size, T is the time series length,
        and C is the number of channels.
        Scaling factor must be already set to self.scale_factor. You can sample it using sample_scaling_factor() or set it manually.

        :param X: Input data of shape (N, T, C).
        :param T_orig: Original time length (T) before scaling.
        :return: Augmented data of shape (N, T_orig, C) with the time dimension rescaled.
        """

        if type(X) == torch.Tensor:
            X = X.detach().cpu().numpy()

        N, T, C = X.shape
        if self.scale_factor > 1.0:
            # In this cale, X is newly sampled in DataLoader, thus X.shape[1] == T_orig * scale_factor
            # Here, make it shorter to match the original length T_orig
            interpolator = PchipInterpolator(np.arange(T), X, axis=1)
            t_new = np.linspace(0, T - 1, T_orig)  # New time points
            X_aug = interpolator(t_new)  # (N, T_orig, C) Matching the original length T

        else:
            # self.scale_factor <= 1.0
            # Cut the first fraction and extend it to match the original length T
            delete_length = int(T * (1.0 - self.scale_factor))
            X_trimmed = X[:, delete_length:, :]  # Trim the first part
            T_new = X_trimmed.shape[1]
            # Stretch the trimmed data to match the original length T
            interpolator = PchipInterpolator(np.arange(T_new), X_trimmed, axis=1)
            new_t = np.linspace(
                0, T_new - 1, T_orig
            )  # New time points for interpolation
            X_aug = interpolator(new_t)  # (N, T_orig, C) Matching the original length T

        return torch.tensor(X_aug, dtype=torch.float32)
