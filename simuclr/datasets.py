import os
from glob import glob

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from .utils import get_default_device, paint


class SensorDataset(Dataset):
    """
    A dataset class for multi-channel time-series data captured by wearable sensors.
    This class is slightly modified from the original implementation at: https://github.com/STRCSussex-UbiCompSiegen/dl_har_public
    """

    def __init__(
        self,
        dataset,
        window,
        stride,
        path_processed,
        stride_test=1,
        prefix=None,
        prefix_wildcard=False,
        verbose=False,
        lazy_load=False,
        scaling="standardize",
        min_vals=None,
        max_vals=None,
        mean=None,
        std=None,
        device=None,
    ):
        """F
        Initialize instance.
        :param dataset: str. Name of target dataset.
        :param window: int. Sliding window size in samples.
        :param stride: int. Step size of the sliding window for training and validation data.
        :param stride_test: int. Step size of the sliding window for testing data.
        :param path_processed: str. Path to directory containing processed training, validation and test data.
        :param prefix: str. Prefix for the filename of the processed data. Options 'train', 'val', or 'test'.
        :param verbose: bool. Whether to print detailed information about the dataset when initializing.
        :param name: str. What to call this dataset (i.e. train, test, val).
        :param lazy_load: bool. Whether to load the whole windowed data into memory or not.
        :param scaling: str. What type of preprocessing to apply to the data. Options 'normalize', 'standardize', or None.
        :param min_vals: numpy array. Minimum values for each sensor channel. Used for normalization.
        :param max_vals: numpy array. Maximum values for each sensor channel. Used for normalization.
        :param mean: numpy array. Mean values for each sensor channel. Used for standardization.
        :param std: numpy array. Standard deviation values for each sensor channel. Used for standardization.
        """

        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.stride_test = stride_test
        self.path_processed = path_processed
        self.verbose = verbose
        self.lazy_load = lazy_load
        self.scaling = scaling
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.mean = mean
        self.std = std

        if device is None:
            self.device = get_default_device()
        else:
            self.device = device

        if prefix is None:
            self.prefix = "No prefix specified"
            self.path_dataset = glob(os.path.join(path_processed, "*.npz"))
        elif isinstance(prefix, str):
            self.prefix = prefix
            self.path_dataset = glob(os.path.join(path_processed, f"{prefix}.npz"))
        elif isinstance(prefix, list):
            self.prefix = prefix
            self.path_dataset = []
            for prefix in prefix:
                self.path_dataset.extend(
                    glob(os.path.join(path_processed, f"{prefix}.npz"))
                )
            print("Loaded the following datasets:")
            print(self.path_dataset)

        self.data = np.concatenate(
            [np.load(path, allow_pickle=True)["data"] for path in self.path_dataset]
        )
        self.target = np.concatenate(
            [np.load(path, allow_pickle=True)["target"] for path in self.path_dataset]
        )

        if self.mean is None:
            self.mean = np.mean(self.data, axis=0)
        if self.std is None:
            self.std = np.std(self.data, axis=0)
            self.std[self.std == 0] = 1
        if self.min_vals is None:
            self.min_vals = np.min(self.data, axis=0)
        if self.max_vals is None:
            self.max_vals = np.max(self.data, axis=0)

        print(f"{self.scaling} is the scaling method used.")
        if self.scaling == "standardize":
            self.data = standardize(self.data, self.mean, self.std)
        elif self.scaling == "normalize":
            self.data = (self.data - self.min_vals) / (self.max_vals - self.min_vals)
        elif self.scaling is None:
            pass
        else:
            raise ValueError(f"Unknown preprocessing scheme {self.scaling}.")

        # To save memory, generate the windowed data on the fly
        if lazy_load:
            # Pre-calculate the number of windows
            self.len = (self.data.shape[0] - self.window) // self.stride + 1
        else:
            self.data, self.target = sliding_window(
                self.data, self.target, self.window, self.stride
            )
            self.len = self.data.shape[0]
            assert self.data.shape[0] == self.target.shape[0]

        print(paint(f"Creating {self.dataset} HAR dataset of size {self.len} ..."))

        self.n_channels = self.data.shape[-1]
        self.n_classes = np.unique(self.target).shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index, scale: float = 1.0):
        """
        Get a single item from the dataset.
        When scale is passed, the window length is scaled by this factor.
        The end of the window is fixed, and the start is adjusted accordingly, to keep the original data in the end.
        :param index:
        :param scale: Only for TimeScaling augmentation
        :return:
        """
        if scale != 1.0 and not self.lazy_load:
            raise ValueError(
                "TimeScaling augmentation is only supported with lazy_load=True."
            )

        if self.lazy_load:
            start = index * self.stride
            end = start + self.window
            target = [int(self._get_label(start, end))]
            if scale != 1.0:
                window_scaled = int(self.window * scale)
                start_scaled = end - window_scaled
                if start_scaled < 0:
                    # Handle the case where start is negative
                    padding_size = abs(start_scaled)

                    # Padding with repeated values at the beginning
                    data = np.pad(
                        self.data[0:end],  # [T, ]
                        ((padding_size, 0), (0, 0)),
                        mode="symmetric",
                    )
                else:
                    data = self.data[start_scaled:end]
            else:
                data = self.data[start:end]
        else:
            data = self.data[index]
            target = [int(self.target[index])]

        data = torch.tensor(data).float()
        target = torch.tensor(target).long()
        idx = torch.from_numpy(np.array(index))
        return data, target, idx

    def _get_label(self, start, end, scheme="max"):
        if scheme == "last":
            return self.target[end - 1]

        elif scheme == "max":
            return np.argmax(np.bincount(self.target[start:end]))

        else:
            raise ValueError(f"Unknown scheme {scheme}.")


def standardize(data, mean=None, std=None, verbose=False):
    """
    Standardizes all sensor channels

    :param data: numpy integer matrix (N, channels)
        Sensor data
    :param mean: (optional) numpy integer array (channels, )
        Array containing mean values for each sensor channel. When given, the mean is subtracted from the data.
    :param std: (optional) numpy integer array (channels, )
        Array containing the standard deviation of each sensor channel. When given, the data is divided by the standard deviation.
    :param verbose: bool
        Whether to print debug information
    :return:
    """
    if verbose:
        # I want to display the mean (given or calculated) here as the verbose message
        if mean is None or std is None:
            print("mean and std not specified. Calculating from data...")
            print(f"data - mean: {np.mean(data, axis=0)}")
            print(f"data - std: {np.std(data, axis=0)}")

    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        std[std == 0] = 1

    if verbose:
        print(f"mean used: {mean}")
        print(f"std used: {std}")

    return (data - mean) / std


def sliding_window(x, y, window, stride, scheme="max"):
    data, target = [], []
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        if scheme == "last":
            # last scheme: : last observed label in the window determines the segment annotation
            y_segment = y[start:end][-1]
        elif scheme == "max":
            # max scheme: most frequent label in the window determines the segment annotation
            y_segment = np.argmax(np.bincount(y[start:end]))
        data.append(x_segment)
        target.append(y_segment)
        start += stride

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int64)

    return data, target
