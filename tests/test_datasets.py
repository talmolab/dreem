"""Test dataset logic."""
import torch
from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.datasets.microscopy_dataset import MicroscopyDataset
from biogtr.datasets.tracking_dataset import TrackingDataset
from torch.utils.data import DataLoader


def test_sleap_dataset(two_flies):
    """Test sleap dataset logic.

    Args:
        two_flies: two flies fixture used for testing
    """
    clip_length = 32

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        crop_type="centroid",
    )

    instances = next(iter(train_ds))

    assert len(instances) == clip_length
    assert len(instances[0]["gt_track_ids"]) == 2
    assert len(instances[0]["gt_track_ids"]) == instances[0]["num_detected"].item()


def test_icy_dataset(ten_icy_particles):
    """Test icy dataset logic.

    Args:
        ten_icy_particles: icy fixture used for testing
    """
    clip_length = 8

    train_ds = MicroscopyDataset(
        videos=[ten_icy_particles[0]],
        tracks=[ten_icy_particles[1]],
        source="icy",
        crop_size=64,
        chunk=True,
        clip_length=clip_length,
    )

    instances = next(iter(train_ds))

    assert len(instances) == clip_length
    assert len(instances[0]["gt_track_ids"]) == 10
    assert len(instances[0]["gt_track_ids"]) == instances[0]["num_detected"].item()


def test_tracking_dataset(two_flies):
    """Test lightning dataset logic.

    Args:
        two_flies: two flies fixture used for testing
    """
    batch_size = 2
    clip_length = 16
    num_workers = 0
    pin_memory = num_workers > 0
    generator = torch.Generator(device="cuda") if torch.cuda.is_available() else None

    train_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        crop_type="centroid",
    )

    train_sleap_dl = DataLoader(
        dataset=train_sleap_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_sleap_ds.no_batching_fn,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=False,
        clip_length=clip_length,
        crop_type="centroid",
    )

    val_sleap_dl = DataLoader(
        dataset=val_sleap_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_sleap_ds.no_batching_fn,
        pin_memory=pin_memory,
        generator=generator,
    )

    test_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=False,
        clip_length=clip_length,
        crop_type="centroid",
    )

    test_sleap_dl = DataLoader(
        dataset=test_sleap_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_sleap_ds.no_batching_fn,
        pin_memory=pin_memory,
        generator=generator,
    )

    # test default if dls are None
    tracking_ds = TrackingDataset(
        train_ds=train_sleap_ds, val_ds=val_sleap_ds, test_ds=test_sleap_ds
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 1
        and tracking_ds.val_dataloader().batch_size == 1
        and tracking_ds.test_dataloader().batch_size == 1
    )

    # test using all dls
    tracking_ds = TrackingDataset(
        train_dl=train_sleap_dl, val_dl=val_sleap_dl, test_dl=test_sleap_dl
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 2
        and tracking_ds.val_dataloader().batch_size == 2
        and tracking_ds.test_dataloader().batch_size == 2
    )

    # test using overrided dls over defaults
    tracking_ds = TrackingDataset(
        train_ds=train_sleap_ds,
        train_dl=train_sleap_dl,
        val_ds=val_sleap_ds,
        val_dl=val_sleap_dl,
        test_ds=test_sleap_ds,
        test_dl=test_sleap_dl,
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 2
        and tracking_ds.val_dataloader().batch_size == 2
        and tracking_ds.test_dataloader().batch_size == 2
    )
    # test using overrided dls over defaults with combinations
    tracking_ds = TrackingDataset(
        train_ds=train_sleap_ds,
        val_ds=val_sleap_ds,
        val_dl=val_sleap_dl,
        test_ds=test_sleap_ds,
        test_dl=test_sleap_dl,
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 1
        and tracking_ds.val_dataloader().batch_size == 2
        and tracking_ds.test_dataloader().batch_size == 2
    )
