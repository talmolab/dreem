"""Module containing classes for lazy loading data."""
from PIL import Image
from typing import Union
import numpy as np
import h5py


class LazyTiffStack:
    """Class used for loading tiffs without loading into memory."""

    def __init__(self, filename: str):
        """Initialize class.

        Args:
            filename: name of tif file to be opened
        """
        # expects spatial, channels
        self.image = Image.open(filename)

    def __getitem__(self, section_idx: int) -> Image:
        """Get frame.

        Args:
            section_idx: index of frame or z-slice to get.

        Returns:
            a PIL image of that frame/z-slice.
        """
        self.image.seek(section_idx)
        return self.image

    def get_section(self, section_idx: int) -> np.array:
        """Get frame as ndarray.

        Args:
            section_idx: index of frame or z-slice to get.

        Returns:
            an np.array of that frame/z-slice.
        """
        section = self.__getitem__(section_idx)
        return np.array(section)

    def close(self):
        """Close tiff stack."""
        self.file.close()


class LazyH5:
    """Class used for loading h5s without loading into memory."""

    def __init__(self, filename: str):
        """Initialize class.

        Args:
            filename: name of tif file to be opened
        """
        # expects spatial, channels
        self.h5 = h5py.File(filename, "r")

    def __getitem__(self, feat: str) -> h5py.Dataset:
        """Get frame.

        Args:
            idx: index of feature to get.

        Returns:
            a h5 group of that feature.
        """
        if not isinstance(feat, str):
            feat = str(feat)
        return self.h5[feat]

    @property
    def keys(self) -> list[str]:
        """Get h5 group keys.

        Returns:
            A list of keys for each group in the h5.
        """
        return list(self.h5.keys())

    def get_section(self, feat, idx) -> np.array:
        """Get frame as ndarray.

        Args:
            idx: index of frame or z-slice to get.

        Returns:
            an np.array of that frame/z-slice.
        """
        feat = self.__getitem__(feat)
        if not isinstance(idx, str):
            idx = str(idx)
        return np.array(feat[idx])

    def close(self):
        """Close h5 file."""
        self.h5.close()
