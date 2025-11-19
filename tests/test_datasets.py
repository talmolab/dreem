"""Test dataset logic."""

import pytest
import torch
from torch.utils.data import DataLoader
import sleap_io as sio
from dreem.datasets import (
    BaseDataset,
    CellTrackingDataset,
    MicroscopyDataset,
    SleapDataset,
    TrackingDataset,
)
from dreem.datasets.data_utils import NodeDropout, get_max_padding


def _prepare_duplicate_instances(dataset, scenario):
    """Placeholder for configuring dataset state prior to iteration.

    Args:
        dataset: `SleapDataset` instance to be customized for the scenario.
        scenario: String key selecting the desired duplicate-detection setup.

    Returns:
        A dictionary containing any metadata required by the tests. Expected keys:
            - 'expected_instance_count': int indicating how many instances should
              remain in the inspected frame after iteration.
            - 'expected_keypoint_counts': list[int] describing the pose sizes that
              should remain after NMS runs.
            - Optional 'removed_keypoint_count': int describing the pose size that
              should be removed by NMS.
            - Optional 'removed_track_ids': list of track ids that should be absent
              after NMS.
            - Optional 'expected_instance_order': list of track ids capturing the
              expected ordering of instances.
            - Optional 'frame_index': int pointing to the frame within the chunk
              that should be inspected (defaults to 0).
    """
    raise NotImplementedError(
        "Implement duplicate instance setup for scenario "
        f"'{scenario}' before running these tests."
    )


def test_base_dataset():
    """Test BaseDataset logic."""

    class DummyDataset(BaseDataset):
        pass

    ds = DummyDataset(
        label_files=[],
        vid_files=[],
        padding=0,
        crop_size=0,
        chunk=False,
        clip_length=0,
        mode="",
    )

    with pytest.raises(NotImplementedError):
        ds.get_indices(0)

    with pytest.raises(NotImplementedError):
        ds.get_instances([], [])

    with pytest.raises(NotImplementedError):
        ds.__getitem__(0)

    with pytest.raises(AttributeError):
        ds.__len__()


def test_sleap_dataset(two_flies):
    """Test sleap dataset logic.

    Args:
        two_flies: two flies fixture used for testing
    """
    clip_length = 32

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        anchors="thorax",
    )

    ds_length = len(train_ds)

    instances = next(iter(train_ds))

    assert len(instances) == clip_length
    assert len(instances[0].get_gt_track_ids()) == 2
    assert len(instances[0].get_gt_track_ids()) == instances[0].num_detected
    assert instances[0].get_crops().shape[1] == 3

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        anchors=["thorax", "head"],
    )

    ds_length = len(train_ds)

    instances = next(iter(train_ds))

    assert instances[0].get_crops().shape[1] == 6

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        anchors=1,
    )

    ds_length = len(train_ds)

    instances = next(iter(train_ds))

    assert instances[0].get_crops().shape[1] == 3

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        anchors=2,
    )

    ds_length = len(train_ds)

    instances = next(iter(train_ds))

    assert instances[0].get_crops().shape[1] == 6

    chunk_frac = 0.5

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        n_chunks=chunk_frac,
    )

    assert len(train_ds) == int(ds_length * chunk_frac)

    n_chunks = 2

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        n_chunks=n_chunks,
    )

    assert len(train_ds) == n_chunks
    assert len(train_ds) == len(train_ds.label_idx)

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        n_chunks=0,
    )

    assert len(train_ds) == ds_length

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        n_chunks=ds_length + 10000,
    )

    assert len(train_ds) == ds_length

    with pytest.raises(Exception):
        train_ds = SleapDataset(
            slp_files=[two_flies[0]],
            video_files=[two_flies[1]],
            data_dirs=["./data/sleap"],
            crop_size=[128, 128],
            chunk=True,
            clip_length=clip_length,
            n_chunks=ds_length + 10000,
        )

    with pytest.raises(Exception):
        train_ds = SleapDataset(
            slp_files=[two_flies[0]],
            video_files=[two_flies[1]],
            data_dirs=["./data/sleap", "./data/microscopy"],
            crop_size=[128],
            chunk=True,
            clip_length=clip_length,
            n_chunks=ds_length + 10000,
        )

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        n_chunks=ds_length + 10000,
    )

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap", "./data/microscopy"],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        n_chunks=ds_length + 10000,
    )

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs="./data/sleap",
        crop_size=128,
        chunk=True,
        max_batching_gap=10,
        clip_length=clip_length,
        n_chunks=30,
    )

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs="./data/sleap",
        crop_size=128,
        chunk=True,
        max_batching_gap=0,
        clip_length=clip_length,
        n_chunks=30,
    )

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs="./data/sleap",
        crop_size=128,
        chunk=False,
        max_batching_gap=10,
        clip_length=clip_length,
        n_chunks=30,
    )

    # test tight bbox implementation
    crop_size = 4
    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs="./data/sleap",
        crop_size=crop_size,
        use_tight_bbox=True,
    )
    frames = next(iter(train_ds))
    for frame in frames:
        for instance in frame.instances:
            _, c, h, w = instance.crop.shape
            assert h <= crop_size and w <= crop_size


def test_sleap_dataset_nms_removes_sparse_duplicate(two_flies_overlapping):
    """High-overlap duplicate with fewer keypoints should be removed."""

    clip_length = 16
    train_ds = SleapDataset(
        slp_files=[two_flies_overlapping[0]],
        video_files=[two_flies_overlapping[1]],
        data_dirs=["./data/sleap"],
        crop_size=70,
        chunk=True,
        clip_length=clip_length,
        anchors=["thorax"],
        max_detection_overlap=0.7,
    )
    frames = next(iter(train_ds))
    frame0 = frames[0]
    frame1 = frames[1]
    assert len(frame0.instances) == 2
    assert len(frame1.instances) == 3
    for instance in frame0.instances:
        assert instance.gt_track_id != -1
        pose_arr = torch.stack([torch.tensor(v) for v in instance.pose.values()])
        assert len(pose_arr[~pose_arr.isnan().any(dim=1)]) > 5


def test_sleap_dataset_nms_retains_instances_below_threshold(two_flies_overlapping):
    """Moderate overlap should not trigger NMS removals."""
    clip_length = 16
    train_ds = SleapDataset(
        slp_files=[two_flies_overlapping[0]],
        video_files=[two_flies_overlapping[1]],
        data_dirs=["./data/sleap"],
        crop_size=70,
        chunk=True,
        clip_length=clip_length,
        anchors=["thorax"],
        max_detection_overlap=0.9,
    )

    frames = next(iter(train_ds))
    frame0 = frames[0]
    frame1 = frames[1]
    assert len(frame0.instances) == 3
    assert len(frame1.instances) == 3


def test_sleap_dataset_nms_off_without_arg(two_flies_overlapping):
    """NMS should be off without max_detection_overlap argument."""
    clip_length = 16
    train_ds = SleapDataset(
        slp_files=[two_flies_overlapping[0]],
        video_files=[two_flies_overlapping[1]],
        data_dirs=["./data/sleap"],
        crop_size=70,
        chunk=True,
        clip_length=clip_length,
        anchors=["thorax"],
        max_detection_overlap=0,
    )

    frames = next(iter(train_ds))
    frame0 = frames[0]
    frame1 = frames[1]
    assert len(frame0.instances) == 3
    assert len(frame1.instances) == 3


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
    assert len(instances[0].get_gt_track_ids()) == 10
    assert len(instances[0].get_gt_track_ids()) == instances[0].num_detected


def test_trackmate_dataset(trackmate_lysosomes):
    """Test trackmate dataset logic.

    Args:
        trackmate_lysosomes: trackmate fixture used for testing
    """
    clip_length = 8

    train_ds = MicroscopyDataset(
        videos=[trackmate_lysosomes[0]],
        tracks=[trackmate_lysosomes[1]],
        source="trackmate",
        crop_size=64,
        chunk=True,
        clip_length=clip_length,
    )

    instances = next(iter(train_ds))

    assert len(instances) == clip_length
    assert len(instances[0].get_gt_track_ids()) == 26
    assert len(instances[0].get_gt_track_ids()) == instances[0].num_detected


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
        assert len(instances[0].get_gt_track_ids()) == num_objects
        assert len(instances[0].get_gt_track_ids()) == instances[0].num_detected


def test_cell_tracking_dataset(cell_tracking):
    """Test cell tracking dataset logic.

    Args:
        cell_tracking: HL60 nuclei fixture used for testing
    """
    clip_length = 8
    raw_img_list, gt_list, ctc_track_meta, data_dir = cell_tracking
    train_ds = CellTrackingDataset(
        raw_img_list=raw_img_list,
        gt_list=gt_list,
        data_dirs=data_dir,
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        ctc_track_meta=ctc_track_meta,
    )
    instances = next(iter(train_ds))

    gt_track_ids_1 = instances[0].get_gt_track_ids()

    assert len(instances) == clip_length
    assert len(gt_track_ids_1) == instances[0].num_detected

    # fall back to using np.unique when gt_list not available
    train_ds = CellTrackingDataset(
        raw_img_list=raw_img_list,
        gt_list=gt_list,
        data_dirs=data_dir,
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        ctc_track_meta=None,
    )

    instances = next(iter(train_ds))

    gt_track_ids_2 = instances[0].get_gt_track_ids()

    assert len(instances) == clip_length
    assert len(gt_track_ids_2) == instances[0].num_detected
    assert gt_track_ids_1.all() == gt_track_ids_2.all()


def test_tracking_dataset(two_flies):
    """Test lightning dataset logic.

    Args:
        two_flies: two flies fixture used for testing
    """
    batch_size = 2
    clip_length = 16
    num_workers = 0
    pin_memory = num_workers > 0

    train_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
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
    )

    val_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
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
    )

    test_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
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

    # test using overridden dls over defaults
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
    # test using overridden dls over defaults with combinations
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
        data_dirs=["./data/sleap"],
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
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=8,
        augmentations=augmentations,
    )

    augs_instances = next(iter(augs_ds))

    a = no_augs_instances[0].get_crops()
    b = augs_instances[0].get_crops()

    assert not torch.all(a.eq(b))

    nodes = ["head", "nose", "tti", "tail"]

    for p in [0, 1]:
        for n in [0, 1, len(nodes)]:
            node_dropout = NodeDropout(p=p, n=n)
            dropped_nodes = node_dropout(nodes)

            if p == 0:
                assert len(dropped_nodes) == 0
            else:
                assert len(dropped_nodes) == n, (
                    f"p={node_dropout.p}, n={node_dropout.n},n_dropped={len(dropped_nodes)}"
                )

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

    a = no_augs_instances[0].get_crops()
    b = augs_instances[0].get_crops()

    assert not torch.all(a.eq(b))


# if __name__ == "__main__":
#     from tests.fixtures.datasets import cell_tracking
#     cell_tracking_args = cell_tracking(("/root/vast/mustafa/dreem-experiments/src/dreem/tests/data/microscopy/cell_tracking"))
#     test_cell_tracking_dataset(cell_tracking_args)
