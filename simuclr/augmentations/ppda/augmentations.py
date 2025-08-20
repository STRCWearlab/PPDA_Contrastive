from typing import List
import torch
from simuclr.utils import get_default_device
from simuclr.augmentations import (
    BaseAugmentation,
    BaseRotation,
    BaseTimeScaling,
    BaseMagnitudeScaling,
    BaseTimeWarping,
    BaseMagnitudeWarping,
    # BaseJittering,
)
from wimusim import WIMUSim
import pytorch3d.transforms.rotation_conversions as rc
import numpy as np
import warnings
from scipy.interpolate import PchipInterpolator, CubicSpline
from pytorch3d.transforms import euler_angles_to_matrix
from simuclr.augmentations import stda


class NoiseBias(BaseAugmentation):
    def __init__(
        self,
        noise_sigma: float = 0.2,
        bias_min: float = -0.2,
        bias_max: float = 0.2,
        device: torch.device = None,
    ):
        if device is None:
            self.device = get_default_device()
        else:
            self.device = device
        self.noise_sigma = noise_sigma
        self.bias_min = bias_min
        self.bias_max = bias_max

    def __str__(self):
        return f"NoiseBias(noise_sigma={self.noise_sigma}, bias_min={self.bias_min}, bias_max={self.bias_max})"

    def sample_bias(self):
        bias_x = torch.FloatTensor(1).uniform_(self.bias_min, self.bias_max)
        bias_y = torch.FloatTensor(1).uniform_(self.bias_min, self.bias_max)
        bias_z = torch.FloatTensor(1).uniform_(self.bias_min, self.bias_max)
        return torch.tensor([bias_x, bias_y, bias_z])  # device=self.device)

    def apply_augmentation(
        self, H: WIMUSim.Hardware, imu_names: List[str] = None
    ) -> WIMUSim.Hardware:
        mu_0 = torch.tensor(
            0.0,
        )  # device=self.device)

        if hasattr(H, "imu_names"):
            imu_names = H.imu_names
        else:
            assert (
                imu_names is not None
            ), "imu_names must be provided when H doesn't have imu_names attribute."

        for imu_name in imu_names:
            sa_sampled = torch.abs(torch.normal(mu_0, self.noise_sigma)).repeat(3)
            sg_sampled = torch.abs(torch.normal(mu_0, self.noise_sigma)).repeat(3)
            ba_sampled = self.sample_bias()
            bg_sampled = self.sample_bias()

            # print(f"sa_sampled: {sa_sampled}")
            # print(f"sg_sampled: {sg_sampled}")
            # print(f"ba_sampled: {ba_sampled}")
            # print(f"bg_sampled: {bg_sampled}")

            H.sa[imu_name] = sa_sampled
            H.sg[imu_name] = sg_sampled
            H.ba[imu_name] = ba_sampled
            H.bg[imu_name] = bg_sampled

        return H


class MagnitudeScaling(BaseMagnitudeScaling):
    def __init__(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        joint_names: List[str],
        mean_center: bool = True,
        scale_translation: bool = False,
        device: torch.device = None,
    ):
        """
        Define the parameters used for scaling the data.

        :param mu:
        :param log_sigma:
        """
        super().__init__(mu, log_sigma, device)
        self.joint_names = joint_names
        self.mean_center = mean_center
        self.scale_translation = scale_translation

    def __str__(self):
        return f"MagnitudeScaling(mu={self.mu}, sigma={torch.exp(self.log_sigma).item()}, joint_names={self.joint_names}, zero_center={self.mean_center}, scale_translation={self.scale_translation})"

    def apply_augmentation(self, D: WIMUSim.Dynamics):
        """
        Apply scaling to specified columns of a (N, T, C) dataset.

        Parameters:
        - X: Input data of shape (N, T, C), where N is the batch size, T is window length and C is the number of channels.
        - sigma: Standard deviation of the Gaussian distribution used for generating the scaling factor.
        - columns: A list of column indices to apply the scaling. If None, applies to all columns.

        Returns:
        - Scaled data with the same shape as the input.
        """

        scale_factor = self.sample_scaling_factors(D.batch_size)

        if self.scale_translation:
            D.translation["XYZ"] = D.translation["XYZ"] * scale_factor[:, None, None]

        for joint_name in self.joint_names:
            D_joint = D.orientation[joint_name]

            axis_angles = rc.quaternion_to_axis_angle(D_joint)  # N, T, 2 (axis angle)
            theta = torch.norm(axis_angles, dim=-1)  # N, T

            if self.mean_center:
                mean_theta = theta.mean(dim=1, keepdim=True)  # [N, 1]
                theta_scaled = (theta - mean_theta) * scale_factor[:, None] + mean_theta
            else:
                theta_scaled = theta * scale_factor[:, None]

            axis_angle_scaled = (
                axis_angles.mT
                / torch.sin(theta / 2)[:, None, :]
                * torch.sin(theta_scaled / 2)[:, None, :]
            ).mT
            D.orientation[joint_name] = rc.axis_angle_to_quaternion(axis_angle_scaled)

        return D


class TimeScaling(BaseTimeScaling):
    def apply_augmentation_v0(
        self, X: torch.Tensor, T_orig: int = None
    ) -> torch.Tensor:
        """
        Apply time scaling to a  (N, J, T, 4) dataset, where N is the batch size, J is the number of joints + 1 for the base tranlation, T is the time series length,
        and C is the number of channels.
        Scaling factor must be already set to self.scale_factor. You can sample it using sample_scaling_factor() or set it manually.

        :param X: Input data of shape  (N, J, T, 4).
        :param T_orig: Original time length (T) before scaling.
        :return: Augmented data of shape (N, J, T, 4) with the time dimension rescaled.
        """
        if type(X) == torch.Tensor:
            X = X.detach().cpu().numpy()

        N, J, T, _ = X.shape  # (N, J, T, 4)
        if self.scale_factor > 1.0:
            # In this cale, X is newly sampled in DataLoader, thus X.shape[1] == T_orig * scale_factor
            # Here, make it shorter to match the original length T_orig
            interpolator = PchipInterpolator(np.arange(T), X, axis=2)
            new_t = np.linspace(0, T - 1, T_orig)  # New time points
            X_aug = interpolator(new_t)  # (N, T_orig, C) Matching the original length T

        else:
            # self.scale_factor <= 1.0
            # Cut the first fraction and extend it to match the original length T
            delete_length = int(T * (1.0 - self.scale_factor))
            X_trimmed = X[:, :, delete_length:, :]  # Trim the first part
            T_new = X_trimmed.shape[2]
            # Stretch the trimmed data to match the original length T
            interpolator = PchipInterpolator(np.arange(T_new), X_trimmed, axis=2)
            new_t = np.linspace(
                0, T_new - 1, T_orig
            )  # New time points for interpolation
            X_aug = interpolator(new_t)  # (N, T_orig, C) Matching the original length T

        return torch.tensor(X_aug, dtype=torch.float32, device=self.device)

    def apply_augmentation(
        self, X: torch.Tensor, T_orig: int = None, use_torch=False
    ) -> torch.Tensor:
        """
        Apply time scaling to a  (N, J, T, 4) dataset, where N is the batch size, J is the number of joints + 1 for the base tranlation, T is the time series length,
        and C is the number of channels.
        Scaling factor must be already set to self.scale_factor. You can sample it using sample_scaling_factor() or set it manually.

        :param X: Input data of shape  (N, J, T, 4).
        :param T_orig: Original time length (T) before scaling.
        :return: Augmented data of shape (N, J, T, 4) with the time dimension rescaled.
        """

        # Convert the data to numpy.
        if use_torch is False and type(X) == torch.Tensor:
            X = X.detach().cpu().numpy()

        N, J, T, _ = X.shape  # (N, J, T, 4)

        if self.scale_factor > 1.0:
            # In this cale, X is newly sampled in DataLoader, thus X.shape[1] == T_orig * scale_factor
            # Here, make it shorter to match the original length T_orig
            new_t = np.linspace(0, T - 1, T_orig)  # New time points

            if use_torch:
                data_interpolator = PchipInterpolator(
                    np.arange(T),
                    X[:, 0, :, :].detach().cpu().numpy(),
                    axis=1,
                    extrapolate=True,
                )
                X_aug = torch.zeros(
                    (N, J, T_orig, 4), dtype=torch.float32, device=self.device
                )
                X_aug[:, 0, :, :] = (
                    torch.from_numpy(data_interpolator(new_t)).float().to(self.device)
                )
                # Interpolate the orientation data
                # (N, J, T, 4)
                X_aug[:, 1:, :, :] = apply_batch_slerp_timescale(
                    X[:, 1:, :, :], new_t, device=self.device
                )
            else:
                data_interpolator = PchipInterpolator(
                    np.arange(T), X[:, 0, :, :], axis=1, extrapolate=True
                )
                X_aug = np.zeros((N, J, T_orig, 4), dtype=X.dtype)
                X_aug[:, 0, :, :] = data_interpolator(new_t)
                X_aug[:, 1:, :, :] = apply_batch_slerp_timescale(X[:, 1:, :, :], new_t)
                X_aug = torch.from_numpy(X_aug).float()

        else:
            # self.scale_factor <= 1.0
            # Cut the first fraction and extend it to match the original length T
            delete_length = int(T * (1.0 - self.scale_factor))
            X_trimmed = X[:, :, delete_length:, :]  # Trim the first part
            T_new = X_trimmed.shape[2]
            # Stretch the trimmed data to match the original length T
            new_t = np.linspace(
                0, T_new - 1, T_orig
            )  # New time points for interpolation

            if use_torch:
                data_interpolator = PchipInterpolator(
                    np.arange(T_new),
                    X_trimmed[:, 0, :, :].detach().cpu().numpy(),
                    axis=1,
                    extrapolate=True,
                )
                X_aug = torch.zeros(
                    (N, J, T_orig, 4), dtype=torch.float32, device=self.device
                )
                X_aug[:, 0, :, :] = (
                    torch.from_numpy(data_interpolator(new_t)).float().to(self.device)
                )
                X_aug[:, 1:, :, :] = apply_batch_slerp_timescale(
                    X_trimmed[:, 1:, :, :], new_t, device=self.device
                )
            else:
                data_interpolator = PchipInterpolator(
                    np.arange(T_new), X_trimmed[:, 0, :, :], axis=1, extrapolate=True
                )
                X_aug = np.zeros((N, J, T_orig, 4), dtype=X.dtype)
                X_aug[:, 0, :, :] = data_interpolator(new_t)
                X_aug[:, 1:, :, :] = torch.tensor(
                    apply_batch_slerp_timescale(X_trimmed[:, 1:, :, :], new_t),
                    dtype=torch.float32,
                )
        return X_aug


class TimeWarping(BaseTimeWarping):
    def apply_augmentation_v0(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply time scaling to a  (N, J, T, 4) dataset, where N is the batch size, J is the number of joints + 1 for the base tranlation, T is the time series length,
        and C is the number of channels.
        Scaling factor must be already set to self.scale_factor. You can sample it using sample_scaling_factor() or set it manually.

        :param X: Input data of shape  (N, J, T, 4).
        :param T_orig: Original time length (T) before scaling.
        :return: Augmented data of shape (N, J, T, 4) with the time dimension rescaled.
        """
        X_type = type(X)
        if X_type == torch.Tensor:
            X = X.detach().cpu().numpy()

        N, J, T, _ = X.shape  # (N, J, T, 4)

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
        warped_indices = warp_interpolator(np.arange(T))  # (T, )

        # Interpolate the original data onto the new, warped timeline for the entire batch
        data_interpolator = PchipInterpolator(np.arange(T), X, axis=2, extrapolate=True)
        X_aug = data_interpolator(warped_indices)

        # Make it a torch tensor
        return torch.tensor(X_aug, dtype=torch.float32, device=self.device)

    def apply_augmentation(self, X: torch.Tensor, use_torch=False) -> torch.Tensor:
        """
        Apply time scaling to a  (N, J, T, 4) dataset, where N is the batch size, J is the number of joints + 1 for the base tranlation, T is the time series length,
        and C is the number of channels.
        Scaling factor must be already set to self.scale_factor. You can sample it using sample_scaling_factor() or set it manually.

        :param X: Input data of shape  (N, J, T, 4).
        :param use_torch: for compatibility with the original implementation. use_torch=False for some of the initial results.
                (though the results should be very close). roughly 25% faster with use_torch=True
        :return: Augmented data of shape (N, J, T, 4) with the time dimension rescaled.
        """
        if use_torch is False and type(X) == torch.Tensor:
            X = X.detach().cpu().numpy()

        N, J, T, _ = X.shape  # (N, J, T, 4)

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
        warped_indices = warp_interpolator(np.arange(T))  # (T, )

        if use_torch:
            X_aug = torch.zeros(X.shape, dtype=torch.float32, device=self.device)

            # The following is the original implementation using PchipInterpolator for translation
            # # Interpolate the translation data
            data_interpolator = PchipInterpolator(
                np.arange(T),
                X[:, 0, :, :].detach().cpu().numpy(),
                axis=1,
                extrapolate=True,
            )
            X_aug[:, 0, :, :] = (
                torch.from_numpy(data_interpolator(warped_indices))
                .float()
                .to(self.device)
            )

            # Translation interpolation (linear, in PyTorch)
            # t = torch.tensor(
            #     warped_indices, device=self.device, dtype=torch.float32
            # )  # (T,)
            # idx0 = torch.floor(t).long().clamp(0, T - 1)
            # idx1 = torch.ceil(t).long().clamp(0, T - 1)
            # alpha = (t - idx0.float()).view(1, 1, -1, 1)  # shape (1, 1, T, 1)
            #
            # X_trans = X[:, 0, :, :]  # (N, T, 4)
            # v0 = X_trans[:, idx0, :]  # (N, T, 4)
            # v1 = X_trans[:, idx1, :]  # (N, T, 4)
            # X_aug[:, 0, :, :] = (1 - alpha) * v0 + alpha * v1
            # (N, J, T, 4)
            X_aug[:, 1:, :, :] = apply_batch_slerp_timewarp(
                X[:, 1:, :, :], warped_indices, device=self.device
            )
        else:
            X_aug = np.zeros(X.shape, dtype=X.dtype)
            # Interpolate the translation data
            data_interpolator = PchipInterpolator(
                np.arange(T), X[:, 0, :, :], axis=1, extrapolate=True
            )
            X_aug[:, 0, :, :] = data_interpolator(warped_indices)

            # Interpolate the orientation data
            # (N, J, T, 4)
            X_aug[:, 1:, :, :] = apply_batch_slerp_timewarp(
                X[:, 1:, :, :], warped_indices, device=None
            )
            X_aug = torch.from_numpy(X_aug).float().to(self.device)
        # Make it a torch tensor
        return X_aug


class MagnitudeWarping(BaseMagnitudeWarping):
    """"""

    # This should be applied to each quaternions.
    # sigma should be somewhere around 0.01 to 0.05 (i.e. 0.1 or 0.2 is too high)
    """"""

    def __init__(
        self,
        sigma: float,
        knot: int,
        mean_center: bool = True,
        device: torch.device = None,
        scale_translation: bool = True,
        joint_idx: list[int] = None,
    ):
        super().__init__(sigma, knot, device)
        self.mean_center = mean_center
        self.scale_translation = scale_translation
        self.joint_idx = joint_idx

    def __str__(self):
        return f"MagnitudeWarping(sigma={self.sigma}, knot={self.knot}, mean_center={self.mean_center}, scale_translation={self.scale_translation})"

    def apply_augmentation(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping to a (N, J, T, 4) dataset using PchipInterpolator.

        Parameters:
        - X:param X: Input data of shape  (N, J, T, 4).

        Returns:
        - Magnitude warped data with the same shape as the input.
        """
        # Convert to numpy if it's a torch tensor
        N, J, T, _ = X.shape  # (N, J, T, 4)

        # Generate common anchor points and magnitude warps for the entire batch
        anchors = np.linspace(0, T - 1, num=self.knot + 2)
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(self.knot + 2,)
        )

        # Generate the magnitude warp indices using PchipInterpolator
        spline_interpolator = CubicSpline(anchors, random_warps, axis=0)
        magnitude_warps = torch.tensor(
            spline_interpolator(np.arange(T)),  # device=self.device
        )  # (T, )

        X_aug = X.detach().clone()

        # Scale the translation by the magnitude warps
        if self.scale_translation:
            X_aug[:, 0, :, :] = X[:, 0, :, :] * magnitude_warps.view(1, 1, T, 1)

        # Calculate the axis angle representation of the quaternions
        axis_angles = rc.quaternion_to_axis_angle(X)  # (N, J, T, 3)
        theta = torch.norm(axis_angles, dim=-1)
        if self.mean_center:
            mean_theta = theta.mean(dim=2, keepdim=True)  # [N, J, 1]
            theta_scaled = (theta - mean_theta) * magnitude_warps.view(
                1, 1, T
            ) + mean_theta
        else:
            theta_scaled = theta * magnitude_warps.view(1, 1, T)

        axis_angle_scaled = (
            axis_angles.mT
            / torch.sin(theta / 2)[:, :, None, :]
            * torch.sin(theta_scaled / 2)[:, :, None, :]
        ).mT
        if self.joint_idx is not None:
            X_aug[:, self.joint_idx, :, :] = rc.axis_angle_to_quaternion(
                axis_angle_scaled[:, self.joint_idx, :, :].float()
            )
        else:
            X_aug[:, 1:, :, :] = rc.axis_angle_to_quaternion(axis_angle_scaled)[
                :, 1:, :, :
            ].float()

        return X_aug


class Rotation(BaseRotation):
    def __init__(
        self,
        rot_range_x: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
        rot_range_y: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
        rot_range_z: torch.Tensor,
        device: torch.device = None,
        flip_prob: float = 0.0,
        paramix: bool = False,
    ):
        super().__init__(rot_range_x, rot_range_y, rot_range_z, device)
        self.flip_prob = flip_prob
        self.paramix = paramix

    def __str__(self):
        (
            min_angle_x,
            max_angle_x,
            min_angle_y,
            max_angle_y,
            min_angle_z,
            max_angle_z,
        ) = self.get_rotation_angles()
        return f"Rotate(rot_range_x=({min_angle_x}, {max_angle_x}), rot_range_y=({min_angle_y}, {max_angle_y}), rot_range_z=({min_angle_z}, {max_angle_z}), flip_prob={self.flip_prob}, paramix={self.paramix})"

    def apply_augmentation(self, P: WIMUSim.Placement) -> WIMUSim.Placement:
        """
        Apply the learnable rotation to 3D data points.

        Parameters:
        - P: WIMUSim.Placement.
          Adjust the orientation of the IMUs with the sampled rotation angles.

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
        rot_angle = torch.tensor([angle_x, angle_y, angle_z])  # , device=self.device)

        # Flip the rotation angles with a certain probability
        if torch.rand(1).item() < self.flip_prob:  # Check if flip should be applied
            # Flip one axis randomly or based on the specified `aug`
            flip_axis = torch.randint(0, 3, (1,)).item()  # Randomly select axis
            rot_angle[flip_axis] += 180.0  # Flip the angle by 180 degrees

        # print(torch.tensor([angle_x, angle_y, angle_z]))
        for imu_name, p_ro in P.ro.items():
            P.ro[imu_name] = p_ro + torch.deg2rad(rot_angle)
        return P


## Util functions for TimeWarping and TimeScaling
def batch_slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Batch Spherical Linear Interpolation (SLERP) for quaternions.
    Args:
        t (np.ndarray or float): Interpolation factor, shape (N, J, T) or scalar.
        v0 (np.ndarray): Starting quaternions, shape (N, J, T, 4).
        v1 (np.ndarray): Target quaternions, shape (N, J, T, 4).
        DOT_THRESHOLD (float): Threshold to switch to linear interpolation.
    Returns:
        np.ndarray: Interpolated quaternions, shape (N, J, T, 4).
    """
    # Ensure v0 and v1 are unit quaternions
    v0 = v0 / np.linalg.norm(v0, axis=-1, keepdims=True)
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)

    # Compute the dot product for each pair of quaternions
    dot = np.sum(v0 * v1, axis=-1, keepdims=True)

    # Clamp dot product to avoid invalid values for arccos
    dot = np.clip(dot, -1.0, 1.0)

    # --- Shortest path correction: flip v1 if dot < 0 ---
    negative_mask = dot < 0.0
    v1 = np.where(negative_mask, -v1, v1)
    dot = np.where(negative_mask, -dot, dot)

    # Use linear interpolation (lerp) for nearly colinear quaternions
    lerp_mask = np.abs(dot) > DOT_THRESHOLD
    result = (1 - t[..., None]) * v0 + t[..., None] * v1

    # For the rest, use SLERP
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Avoid division by zero in SLERP
    sin_theta_0 = np.where(sin_theta_0 == 0, 1e-8, sin_theta_0)

    theta_t = theta_0 * t[..., None]
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0
    slerp_result = s0 * v0 + s1 * v1

    # Combine lerp and slerp results based on the mask
    result = np.where(lerp_mask, result, slerp_result)

    return result


def batch_slerp_torch(
    t: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD=0.9995
) -> torch.Tensor:
    """
    Batched SLERP (Spherical Linear Interpolation) for quaternions in PyTorch.

    Args:
        t (torch.Tensor): Interpolation factors, shape (T,) or (N, J, T).
        v0 (torch.Tensor): Start quaternions, shape (N, J, T, 4).
        v1 (torch.Tensor): End quaternions, shape (N, J, T, 4).
        DOT_THRESHOLD (float): Threshold above which linear interpolation is used instead of SLERP.

    Returns:
        torch.Tensor: Interpolated quaternions, shape (N, J, T, 4).
    """
    # Normalize input quaternions
    v0 = v0 / v0.norm(dim=-1, keepdim=True)
    v1 = v1 / v1.norm(dim=-1, keepdim=True)

    # Dot product between v0 and v1
    dot = (v0 * v1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    # --- Shortest path correction: flip v1 if dot < 0 ---
    v1 = torch.where(dot < 0.0, -v1, v1)
    dot = dot.abs()

    # Create mask for near-identical quaternions
    lerp_mask = (dot.abs() > DOT_THRESHOLD).squeeze(-1)

    # Interpolation factor: shape handling
    if t.ndim == 1:
        t = t.view(1, 1, -1, 1)  # broadcast to (N, J, T, 1)
    elif t.ndim == 3:
        t = t.unsqueeze(-1)  # ensure shape (N, J, T, 1)
    else:
        raise ValueError("t must be shape (T,) or (N, J, T)")

    # Linear interpolation
    lerp = (1.0 - t) * v0 + t * v1

    # SLERP interpolation
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp(min=1e-8)

    theta_t = theta_0 * t
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    slerp = s0 * v0 + s1 * v1

    # Combine SLERP and LERP using mask
    return torch.where(lerp_mask.unsqueeze(-1), lerp, slerp)


def apply_batch_slerp_timewarp(X, new_indices, DOT_THRESHOLD=0.9995, device=None):
    """
    Applies SLERP-based time-warping to quaternion data for joints, excluding translation.

    Args:
        X (torch.Tensor): Input data of shape (N, J, T, 4).
        new_indices (np.ndarray): Warped time indices, shape (T,).
        DOT_THRESHOLD (float): Threshold for considering vectors as colinear.

    Returns:
        torch.Tensor: Time-warped data of shape (N, J, T, 4).
    """
    is_torch = isinstance(X, torch.Tensor)
    if is_torch:
        device = X.device
        X = X.detach().cpu().numpy()

    N, J, T, C = X.shape

    # Clamp idx0 and idx1 to be within bounds [0, T-1]
    idx0 = np.clip(np.floor(new_indices).astype(int), 0, T - 1)
    idx1 = np.clip(np.ceil(new_indices).astype(int), 0, T - 1)

    # Define v0 and v1 for the full T steps (start with initial frame as v0)
    v0 = np.take_along_axis(X, idx0[None, None, :, None], axis=2)
    v1 = np.take_along_axis(X, idx1[None, None, :, None], axis=2)
    # Calculate interpolation factors `t` between consecutive warped indices for T-1 steps
    # t = np.interp(np.arange(T), new_indices, np.arange(T)) / (T - 1)
    idx0 = np.floor(new_indices).astype(int)
    idx1 = np.clip(np.ceil(new_indices).astype(int), 0, T - 1)  # Clamped to bounds
    t = (new_indices - idx0) / np.maximum(idx1 - idx0, 1)
    # Initialize an array to hold the augmented quaternions
    if device is None:
        X_quat_aug = batch_slerp(t, v0, v1, DOT_THRESHOLD=DOT_THRESHOLD)
    else:
        X_quat_aug = batch_slerp_torch(
            torch.Tensor(t).to(device),
            torch.Tensor(v0).to(device),
            torch.Tensor(v1).to(device),
            DOT_THRESHOLD=DOT_THRESHOLD,
        )
    # print("Using numpy for SLERP")
    # X_aug = batch_slerp(t, v0, v1, DOT_THRESHOLD=DOT_THRESHOLD)

    return X_quat_aug


def apply_batch_slerp_timescale(X, new_t, DOT_THRESHOLD=0.9995, device=None):
    """
    SLERP for time-scaled data. Interpolates quaternion data from length T_scaled to T_orig.

    Args:
        X (np.ndarray): Input quaternion data, shape (N, J-1, T_scaled, 4).
        new_t (np.ndarray): New time points to rescale data to shape (T_orig,).
        DOT_THRESHOLD (float): Threshold to switch to linear interpolation.

    Returns:
        np.ndarray: Interpolated quaternion data, shape (N, J-1, T_orig, 4).
    """

    is_torch = isinstance(X, torch.Tensor)
    if is_torch:
        X = X.detach().cpu().numpy()

    N, J, T_scaled, C = X.shape

    T_orig = len(new_t)
    # Initialize the output array
    X_quat_aug = np.zeros((N, J, T_orig, C), dtype=X.dtype)

    # Set up indices for original and target time steps
    idx0 = np.clip(np.floor(new_t).astype(int), 0, T_scaled - 1)
    idx1 = np.clip(np.ceil(new_t).astype(int), 0, T_scaled - 1)

    # Select v0 and v1 quaternions based on the indices
    v0 = np.take_along_axis(X, idx0[None, None, :, None], axis=2)
    v1 = np.take_along_axis(X, idx1[None, None, :, None], axis=2)

    # Calculate interpolation factor for each target frame
    t = (new_t - idx0) / np.maximum(idx1 - idx0, 1)
    # t = t[None, None, :, None]  # Shape (1, 1, T_orig, 1) for broadcasting

    # Perform SLERP interpolation using `t` between `v0` and `v1`
    if device is None:
        X_quat_aug = batch_slerp(t, v0, v1, DOT_THRESHOLD=DOT_THRESHOLD)
    else:
        X_quat_aug = batch_slerp_torch(
            torch.Tensor(t).to(device),
            torch.Tensor(v0).to(device),
            torch.Tensor(v1).to(device),
            DOT_THRESHOLD=DOT_THRESHOLD,
        )
    return X_quat_aug
