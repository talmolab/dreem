import torch
from pytorch_lightning import LightningDataModule
from typing import Union
from torch.utils.data import DataLoader
from biogtr.datasets.microscopy_dataset import MicroscopyDataset
from biogtr.datasets.sleap_dataset import SleapDataset

# todo: move to config
num_workers = 0
shuffle = True

device = "cuda" if torch.cuda.is_available() else "cpu"

if num_workers > 0:
    # prevent too many open files error
    pin_memory = True
    torch.multiprocessing.set_sharing_strategy("file_system")
else:
    pin_memory = False

# for dataloader if shuffling, since shuffling is done by default on cpu
generator = torch.Generator(device=device) if shuffle else None

# useful for longer training runs, but not for single iteration debugging
# finds optimal hardware algs which has upfront time increase for first
# iteration, quicker subsequent iterations

# torch.backends.cudnn.benchmark = True

# pytorch 2 logic - we set our device once here so we don't have to keep setting
torch.set_default_device(device)

"""
Lightning wrapper for tracking datasets
"""


class TrackingDataset(LightningDataModule):
    def __init__(
        self,
        train_ds: Union[SleapDataset, MicroscopyDataset, None] = None,
        train_dl: DataLoader = None,
        val_ds: Union[SleapDataset, MicroscopyDataset, None] = None,
        val_dl: DataLoader = None,
        test_ds: Union[SleapDataset, MicroscopyDataset, None] = None,
        test_dl: DataLoader = None,
    ):
        """Initialize tracking dataset
        Args:
            train_ds: Sleap or Microscopy training Dataset
            train_dl: Training dataloader. Only used for overriding `train_dataloader`.
            val_ds: Sleap or Microscopy Validation set
            val_dl : Validation dataloader. Only used for overriding `val_dataloader`.
        """
        # assert (
        #     train_ds is not None or train_dl is not None
        # ), "Must pass in either a train dataset or train dataloader"
        # assert (
        #     val_ds is not None or val_dl is not None
        # ), "Must pass in either a val dataset or val dataset"
        # assert (
        #     test_ds is not None or test_dl is not None
        # ), "Must pass in either a test dataset or test dataset"
        super().__init__()
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.val_ds = val_ds
        self.val_dl = val_dl
        self.test_ds = test_ds
        self.test_dl = test_dl

    def setup(self, stage=None):
        pass

    def train_dataloader(self) -> DataLoader:
        """Getter for train_dataloader.
        Returns: The Training Dataloader.
        """
        if self.train_dl is None:
            return DataLoader(
                self.train_ds,
                batch_size=1,
                shuffle=True,
                pin_memory=False,
                collate_fn=self.train_ds.no_batching_fn,
                num_workers=0,
                generator=torch.Generator(device="cuda")
                if torch.cuda.is_available()
                else None,
            )
        else:
            return self.train_dl

    def val_dataloader(self) -> DataLoader:
        """Getter for val dataloader
        Returns: The validation dataloader.
        """
        if self.val_dl is None:
            return DataLoader(
                self.val_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=False,
                collate_fn=self.train_ds.no_batching_fn,
                num_workers=0,
                generator=None,
            )
        else:
            return self.val_dl

    def test_dataloader(self) -> DataLoader:
        """Getter for test dataloader
        Returns: The test dataloader
        """
        if self.test_dl is None:
            return DataLoader(
                self.test_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=pin_memory,
                collate_fn=self.train_ds.no_batching_fn,
                num_workers=num_workers,
                generator=None,
            )
        else:
            return self.test_dl
