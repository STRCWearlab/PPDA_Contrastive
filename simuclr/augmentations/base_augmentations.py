import numpy as np
import torch
import warnings
from simuclr.utils import get_default_device


class BaseAugmentation:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Allows the object to be called directly as a function.
        """
        return self.apply_augmentation(X)

    def apply_augmentation(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Identity(BaseAugmentation):
    """
    A dummy augmentation class that does not apply any augmentation.

    """

    def __init__(self, device):
        self.device = device
        pass

    def apply_augmentation(self, X: torch.Tensor):
        return X

    def __str__(self):
        return f"Identity(device={self.device})"


class BaseRotation(BaseAugmentation):
    def __init__(
        self,
        rot_range_x: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
        rot_range_y: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
        rot_range_z: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
        device: torch.device = None,
    ):
        """
        Define the parameters used for rotating the data.

        :param rot_range_x: Range of rotation angles around the x-axis.
        :param rot_range_y: Range of rotation angles around the y-axis.
        :param rot_range_z: Range of rotation angles around the z-axis.
        :param device: Torch device (CPU/GPU).
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize learnable parameters for each axis based on the input tensor
        # We use logit (inverse sigmoid) to initialize the values based on the desired range
        self.min_angle_param_x = torch.logit((rot_range_x[0] + 180) / 360)
        self.angle_range_param_x = torch.logit(
            (rot_range_x[1] - rot_range_x[0]) / (180 - rot_range_x[0])
        )

        self.min_angle_param_y = torch.logit((rot_range_y[0] + 180) / 360)
        self.angle_range_param_y = torch.logit(
            (rot_range_y[1] - rot_range_y[0]) / (180 - rot_range_y[0])
        )

        self.min_angle_param_z = torch.logit((rot_range_z[0] + 180) / 360)
        self.angle_range_param_z = torch.logit(
            (rot_range_z[1] - rot_range_z[0]) / (180 - rot_range_z[0])
        )

        # Scaling factor for (-180, 180) degrees
        self.angle_scale = 360.0

    def __str__(self):
        (
            min_angle_x,
            max_angle_x,
            min_angle_y,
            max_angle_y,
            min_angle_z,
            max_angle_z,
        ) = self.get_rotation_angles()
        return f"Rotate(rot_range_x=({min_angle_x}, {max_angle_x}), rot_range_y=({min_angle_y}, {max_angle_y}), rot_range_z=({min_angle_z}, {max_angle_z}))"

    def get_rotation_angles(self):
        """
        Calculate the min_angle and max_angle for each axis (x, y, z) based on learnable parameters.
        Ensures that min_angle < max_angle and all angles are within (-180, 180).
        """
        # X-axis angles
        min_angle_x = -180 + torch.sigmoid(self.min_angle_param_x) * self.angle_scale
        max_angle_x = min_angle_x + torch.sigmoid(self.angle_range_param_x) * (
            180 - min_angle_x
        )

        # Y-axis angles
        min_angle_y = -180 + torch.sigmoid(self.min_angle_param_y) * self.angle_scale
        max_angle_y = min_angle_y + torch.sigmoid(self.angle_range_param_y) * (
            180 - min_angle_y
        )

        # Z-axis angles
        min_angle_z = -180 + torch.sigmoid(self.min_angle_param_z) * self.angle_scale
        max_angle_z = min_angle_z + torch.sigmoid(self.angle_range_param_z) * (
            180 - min_angle_z
        )

        return (
            min_angle_x,
            max_angle_x,
            min_angle_y,
            max_angle_y,
            min_angle_z,
            max_angle_z,
        )


class BaseMagnitudeScaling(BaseAugmentation):
    """ """

    def __init__(
        self, mu: torch.Tensor, log_sigma: torch.Tensor, device: torch.device = None
    ):
        """
        Define the parameters used for scaling the data.

        :param mu:
        :param log_sigma:
        """

        self.mu = mu
        self.log_sigma = log_sigma

        if device is None:
            self.device = get_default_device()
        else:
            self.device = device

    def __str__(self):
        return (
            f"MagnitudeScaling(mu={self.mu}, sigma={torch.exp(self.log_sigma).item()})"
        )

    def sample_scaling_factors(self, batch_size, cpu=True) -> torch.Tensor:
        """
        Sample a scaling factor from the Gaussian distribution defined by mu and sigma.

        :return: sigma to be used for scaling the sensor data
        """
        if cpu:
            epsilon = torch.randn(batch_size)
        else:
            epsilon = torch.randn(batch_size, device=self.device)
        sigma = torch.exp(self.log_sigma)
        # print(self.mu.device, sigma.device, epsilon.device)
        return self.mu + sigma * epsilon


class BaseTimeScaling(BaseAugmentation):
    def __init__(self, scale_factor_min: float, scale_factor_max: float):  # device):
        """

        **This augmentation does not support parameter optimization**
        :param scale_factor_min:
        :param scale_factor_max:
        :param device:
        """

        # self.device = device or torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        if isinstance(scale_factor_min, torch.Tensor) or isinstance(
            scale_factor_max, torch.Tensor
        ):
            warnings.warn(
                "Parameter scale_factor_min and scale_factor_max are not learnable as this augmentation is implemented in numpy."
            )
            self.scale_factor_min = float(scale_factor_min.detach().cpu())
            self.scale_factor_max = float(scale_factor_max.detach().cpu())
        elif isinstance(scale_factor_min, float) and isinstance(
            scale_factor_max, float
        ):
            self.scale_factor_min = scale_factor_min
            self.scale_factor_max = scale_factor_max
        else:
            raise ValueError(
                "scale_factor_min and scale_factor_max must be either torch.Tensor or float."
            )

        self.scale_factor = (
            scale_factor_max - scale_factor_min
        ) * 0.5 + scale_factor_min

    def __str__(self):
        return f"TimeScaling(scale_min={self.scale_factor_min}, scale_max={self.scale_factor_max})"

    def sample_scaling_factor(self):
        self.scale_factor = np.random.uniform(
            low=self.scale_factor_min, high=self.scale_factor_max
        )
        return self.scale_factor

    @property
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, scale_factor):
        self._scale_factor = scale_factor


class BaseTimeWarping(BaseAugmentation):
    def __init__(
        self,
        sigma: float = 0.1,
        knot: int = 4,
        max_speed_ratio: float = 1.5,
        device: torch.device = None,
    ):
        """
        TimeWarping augmentation for time series data.

        :param sigma: The amount of warping (higher values cause more time distortion).
        :param knot: Number of warping control points (splits the time axis into segments).
        :param max_speed_ratio: Maximum speed ratio for time warping.
        :param device: Torch device (CPU/GPU).
        """
        self.sigma = sigma  # Currently this sigma is not used
        self.knot = knot
        self.max_speed_ratio = max_speed_ratio  # Maximum speed ratio for time warping

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        warnings.warn(
            "Parameter sigma is not learnable as this augmentation is implemented in numpy."
        )

    def __str__(self):
        return f"TimeWarping(sigma={self.sigma}, knot={self.knot}, max_speed_ratio={self.max_speed_ratio})"


class BaseMagnitudeWarping(BaseAugmentation):
    def __init__(
        self,
        sigma: float = 0.1,
        knot: int = 4,
        device: torch.device = None,
    ):
        """
        MagnitudeWarping augmentation for time series data.

        :param sigma: The amount of warping (higher values cause more time distortion).
        :param knot: Number of warping control points (splits the time axis into segments).
        :param device: Torch device (CPU/GPU).
        """

        self.sigma = sigma  # learnable parameter (if require_gradient is True)
        self.knot = knot

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        warnings.warn(
            "Parameter sigma is not learnable as this augmentation is implemented in numpy."
        )

    def __str__(self):
        return f"MagnitudeWarping(sigma={self.sigma}, knot={self.knot})"


class BaseJittering(BaseAugmentation):
    def __init__(self, log_sigma: torch.Tensor, device: torch.device = None):
        """
        Define the parameters used for applying jittering
        :param log_sigma (torch.Tensor): use log_sigma to make it positive
        """
        self.log_sigma = log_sigma

        if device is None:
            self.device = get_default_device()
        else:
            self.device = device

    def __str__(self):
        return f"Jittering(sigma={torch.exp(self.log_sigma).item()})"

    def sample_noise(self, batch_shape) -> torch.Tensor:
        """
        Sample a noise factor from the Gaussian distribution defined by mu (=0.) and sigma.

        :return: sigma to be used for scaling the sensor data
        """
        # noise = torch.randn(batch_shape, device=self.device) * torch.exp(self.log_sigma)
        noise = torch.randn(batch_shape) * torch.exp(self.log_sigma)
        return noise  # mean of 0 and std of sigma
