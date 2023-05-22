"""Test dataset logic."""
from biogtr.datasets.data_utils import get_max_padding
from biogtr.datasets.microscopy_dataset import MicroscopyDataset
from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.datasets.tracking_dataset import TrackingDataset
from torch.utils.data import DataLoader
import torch


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


def test_isbi_dataset(isbi_microtubules, isbi_receptors):
    """Test isbi dataset logic.

    Args:
        isbi_microtubules: isbi microtubules fixture used for testing
        isbi_receptors: isbi receptors fixture used for testing
    """

    clip_length = 8

    for ds in [isbi_microtubules, isbi_receptors]:
        num_objects = 47 if ds == isbi_microtubules else 61

        train_ds = MicroscopyDataset(
            videos=[ds[0]],
            tracks=[ds[1]],
            source="isbi",
            crop_size=64,
            chunk=True,
            clip_length=clip_length,
        )

        instances = next(iter(train_ds))

        assert len(instances) == clip_length
        assert len(instances[0]["gt_track_ids"]) == num_objects
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


def test_augmentations(two_flies, ten_icy_particles):
    """Test augmentations.

    Args:
        two_flies: flies fixture used for testing
        ten_icy_particles: icy fixture used for testing
    """

    no_augs_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=8,
    )

    no_augs_instances = next(iter(no_augs_ds))

    augmentations = {
        "Rotate": {"limit": 45, "p": 1.0},
        "GaussianBlur": {"blur_limit": (3, 7), "sigma_limit": 0, "p": 1.0},
        "GaussNoise": {
            "var_limit": (10.0, 50.0),
            "mean": 0,
            "per_channel": True,
            "p": 1.0,
        },
    }

    augs_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=8,
        augmentations=augmentations,
    )

    augs_instances = next(iter(augs_ds))

    a = no_augs_instances[0]["crops"]
    b = augs_instances[0]["crops"]

    assert not torch.all(a.eq(b))

    no_augs_ds = MicroscopyDataset(
        videos=[ten_icy_particles[0]],
        tracks=[ten_icy_particles[1]],
        source="icy",
        crop_size=64,
        chunk=True,
        clip_length=8,
    )

    no_augs_instances = next(iter(no_augs_ds))

    # pad before rotation
    padded_height, padded_width = get_max_padding(512, 512)

    pad = {
        "PadIfNeeded": {
            "min_height": padded_height,
            "min_width": padded_width,
            "border_mode": 0,
        }
    }

    pad.update(augmentations)
    augmentations = pad

    # add another to end
    motion = {"MotionBlur": {"blur_limit": (3, 7), "p": 0.5}}

    augmentations.update(motion)

    augs_ds = MicroscopyDataset(
        videos=[ten_icy_particles[0]],
        tracks=[ten_icy_particles[1]],
        source="icy",
        crop_size=64,
        chunk=True,
        clip_length=8,
        augmentations=augmentations,
    )

    augs_instances = next(iter(augs_ds))

    a = no_augs_instances[0]["crops"]
    b = augs_instances[0]["crops"]

    assert not torch.all(a.eq(b))
