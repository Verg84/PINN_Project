import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from torch.utils.data import Dataset


class TempDataset(Dataset):
    """
    Input:
        Coarse (downsampled + upsampled) sequence

    Target:
        Original high-resolution field

    Mask:
        0 = coarse-grid pixels (ignore)
        1 = held-out pixels (supervise)
    """

    def __init__(self, X, y, mask):
        self.X = X.float()
        self.y = y.float()
        self.mask = mask.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask


##########################################################################


class TempDataModule:

    def __init__(
        self,
        nc_file,
        sequence_length=3,
        train_ratio=0.70,
        val_ratio=0.15,
        downsample_factor=None,
    ):

        self.nc_file = nc_file
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.downsample_factor = downsample_factor

        self.mean = None
        self.std = None

        self.lat = None
        self.lon = None

    ######################################################################

    def load(self):

        with xr.open_dataset(self.nc_file) as ds:

            t2m = ds["t2m"].values.astype(np.float32)

            self.lat = ds["lat"].values
            self.lon = ds["lon"].values

        ##############################################################
        # Create coarse-resolution inputs
        ##############################################################

        if self.downsample_factor is not None:

            factor = self.downsample_factor

            x = torch.from_numpy(t2m).unsqueeze(1)

            original_size = x.shape[-2:]

            # Downsample
            x = F.avg_pool2d(
                x,
                kernel_size=factor,
                stride=factor,
            )

            # Upsample back to original resolution
            x = F.interpolate(
                x,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

            coarse = x.squeeze(1).numpy()

            ##########################################################
            # Supervision mask
            ##########################################################

            H, W = original_size

            mask = np.ones((H, W), dtype=np.float32)

            # Pixels that belong to the coarse grid
            mask[::factor, ::factor] = 0.0

        else:

            coarse = t2m.copy()

            H, W = t2m.shape[1:]

            # Supervise every pixel if no downsampling
            mask = np.ones((H, W), dtype=np.float32)

        ##############################################################
        # Normalize
        ##############################################################

        n_train = int(len(t2m) * self.train_ratio)

        self.mean = t2m[:n_train].mean()
        self.std = t2m[:n_train].std()

        coarse = (coarse - self.mean) / self.std
        t2m = (t2m - self.mean) / self.std

        ##############################################################
        # Build sequences
        ##############################################################

        X = []
        y = []

        seq = self.sequence_length

        for i in range(len(t2m) - seq):

            X.append(coarse[i:i + seq])

            y.append(t2m[i + seq])

        X = torch.from_numpy(np.asarray(X)).float()

        y = torch.from_numpy(
            np.asarray(y)
        ).unsqueeze(1).float()

        ##############################################################
        # Mask tensor
        ##############################################################

        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return X, y, mask

    ######################################################################

    def split(self):

        X, y, mask = self.load()

        N = len(X)

        train_end = int(self.train_ratio * N)

        val_end = int(
            (self.train_ratio + self.val_ratio) * N
        )

        train_dataset = TempDataset(
            X[:train_end],
            y[:train_end],
            mask,
        )

        val_dataset = TempDataset(
            X[train_end:val_end],
            y[train_end:val_end],
            mask,
        )

        test_dataset = TempDataset(
            X[val_end:],
            y[val_end:],
            mask,
        )

        return train_dataset, val_dataset, test_dataset

    ######################################################################

    def denormalize(self, tensor):
        return tensor * self.std + self.mean

    ######################################################################

    def get_statistics(self):
        return self.mean, self.std

    ######################################################################

    def get_coordinates(self):
        return self.lat, self.lon