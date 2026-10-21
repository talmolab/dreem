"""Test dataset logic."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dreem.datasets import (
    BaseDataset,
    CellTrackingDataset,
    MicroscopyDataset,
    SleapDataset,
    TrackingDataset,
)
from dreem.datasets.data_utils import (
    ClipAugmentation,
    NodeDropout,
    build_augmentations,
    crop_bbox,
    get_bbox,
    get_max_padding,
)


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


def test_cell_tracking_dataset_no_chunk(cell_tracking):
    """Test CellTrackingDataset with chunk=False doesn't crash.

    Regression test: create_chunks_other() previously accessed self.labels
    which is None for CTC datasets. Should use self.frame_idx instead.

    Args:
        cell_tracking: HL60 nuclei fixture used for testing
    """
    raw_img_list, gt_list, ctc_track_meta, data_dir = cell_tracking
    ds = CellTrackingDataset(
        raw_img_list=raw_img_list,
        gt_list=gt_list,
        data_dirs=data_dir,
        crop_size=128,
        chunk=False,
        clip_length=8,
        ctc_track_meta=ctc_track_meta,
    )

    # chunk=False: one entry per dataset, all frames in one batch
    assert len(ds) == 1
    assert len(ds.label_idx) == 1
    assert ds.label_idx[0] == 0
    assert len(ds.chunked_frame_idx[0]) == len(ds.frame_idx[0])


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


@pytest.mark.parametrize(
    "center",
    [
        (500.0, 400.0),  # interior
        (10.0, 20.0),  # near the left and top edges
        (-5.0, 3.0),  # anchor just outside the frame
        (1020.0, 700.0),  # near the right edge of a 1024-wide frame
    ],
)
def test_get_bbox_is_centered(center):
    """The box is centered on the anchor even when it extends past the frame.

    Args:
        center: (x, y) anchor position in pixels
    """
    y1, x1, y2, x2 = get_bbox(center, 128).tolist()

    assert (x2 - x1, y2 - y1) == (128, 128)
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == center


def test_crop_near_edge_is_centered():
    """A crop of an instance near the top-left corner is centered on it, zero-padded."""
    img = torch.zeros(1, 100, 100)
    img[0, 20, 10] = 1.0  # instance at x=10, y=20

    crop = crop_bbox(img, get_bbox((10.0, 20.0), 32))

    assert crop.shape == (1, 32, 32)
    assert crop[0, 16, 16] == 1.0
    assert (crop[0, :, :6] == 0).all()  # columns left of the frame are padding


def test_spatial_augmentations_consistent_across_clip(two_flies):
    """Spatial augmentations must use one set of parameters for a whole clip.

    Drawing a rotation per frame makes the same fly jump between consecutive
    frames of a training clip, which never happens at inference.

    Args:
        two_flies: flies fixture used for testing
    """
    ds_args = dict(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        anchors="centroid",
        crop_size=128,
        chunk=True,
        clip_length=8,
    )
    raw_clip = next(iter(SleapDataset(**ds_args)))
    aug_clip = next(
        iter(
            SleapDataset(
                **ds_args,
                augmentations={
                    "Rotate": {"limit": (20, 45), "p": 1.0},
                    "GaussianBlur": {"blur_limit": (3, 7), "p": 1.0},
                },
            )
        )
    )

    def fly_to_fly(frame):
        """Vector from track 0's centroid to track 1's (instance order is shuffled)."""
        centroids = {
            instance.gt_track_id.item(): np.asarray(instance.centroid["centroid"])
            for instance in frame.instances
        }
        if not {0, 1} <= centroids.keys():
            return None
        return centroids[1] - centroids[0]

    # The rotation angle is recovered from the vector between the two flies, which
    # does not depend on the center of rotation.
    angles = []
    for raw_frame, aug_frame in zip(raw_clip, aug_clip):
        raw_vec, aug_vec = fly_to_fly(raw_frame), fly_to_fly(aug_frame)
        if raw_vec is None or aug_vec is None:
            continue
        cross = raw_vec[0] * aug_vec[1] - raw_vec[1] * aug_vec[0]
        angles.append(np.arctan2(cross, np.dot(raw_vec, aug_vec)))

    assert len(angles) >= 2
    assert np.ptp(angles) < 1e-3, f"rotation varies within a clip: {angles}"
    assert abs(angles[0]) > np.deg2rad(15)


def test_sleap_dataset_video_readers_are_per_process(two_flies):
    """A process that inherits a dataset opens its own readers and leaves the parent's.

    Args:
        two_flies: flies fixture used for testing
    """
    ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=8,
    )
    assert ds.videos == [None]  # nothing is opened until a frame is read

    ds[0]
    parent_reader = ds.videos[0]
    assert parent_reader is not None

    ds._videos_pid = -1  # as a forked DataLoader worker sees the parent's state
    ds[0]
    assert ds.videos[0] is not parent_reader
    assert not parent_reader.closed  # closing it would kill the parent's ffmpeg
    parent_reader.get_data(0)


def test_sleap_dataset_multiworker_crops_match(two_flies):
    """Crops loaded by DataLoader workers match those loaded in the main process.

    Args:
        two_flies: flies fixture used for testing
    """
    ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        data_dirs=["./data/sleap"],
        crop_size=128,
        chunk=True,
        clip_length=8,
    )

    def crops_by_instance(num_workers):
        """Map (frame id, track id) to crop; instance order is shuffled per frame."""
        loader = DataLoader(
            ds, batch_size=1, collate_fn=ds.no_batching_fn, num_workers=num_workers
        )
        return {
            (frame.frame_id.item(), instance.gt_track_id.item()): instance.crop
            for batch in loader
            for frame in batch[0]
            for instance in frame.instances
        }

    single_process = crops_by_instance(num_workers=0)
    workers = crops_by_instance(num_workers=2)

    assert single_process.keys() == workers.keys()
    mismatched = [
        key
        for key in single_process
        if not torch.equal(single_process[key], workers[key])
    ]
    assert not mismatched, f"crops differ for (frame, track) {mismatched}"


def _fit_rigid(src, dst):
    """Fit `dst ~= src @ M.T + t` and return (rotation angle in degrees, flipped).

    Args:
        src: (n, 2) array of points before augmentation.
        dst: (n, 2) array of the same points after augmentation.

    Returns:
        The rotation angle of the fitted linear map (after undoing any flip) and
        whether it contains a reflection.
    """
    X = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    M = sol[:2].T
    flipped = np.linalg.det(M) < 0
    if flipped:
        M = M @ np.diag([-1, 1])
    return np.degrees(np.arctan2(M[1, 0], M[0, 0])), flipped


def _bbox_centers(frame):
    """Map gt track id -> bbox center (x, y) for every instance in a frame."""
    out = {}
    for inst in frame.instances:
        ymin, xmin, ymax, xmax = inst.bbox.reshape(-1, 4)[0].tolist()
        out[int(inst.gt_track_id)] = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    return out


def _clip_transforms(raw_frames, aug_frames, get_points):
    """Fit the raw -> augmented point transform for each frame of a clip."""
    transforms = []
    for raw_frame, aug_frame in zip(raw_frames, aug_frames):
        raw, aug = get_points(raw_frame), get_points(aug_frame)
        keys = [
            k
            for k in raw
            if k in aug and np.isfinite(raw[k]).all() and np.isfinite(aug[k]).all()
        ]
        assert len(keys) >= 3
        transforms.append(
            _fit_rigid(
                np.array([raw[k] for k in keys]), np.array([aug[k] for k in keys])
            )
        )
    return transforms


def _assert_same_transform(transforms, atol):
    angles = np.array([angle for angle, _ in transforms])
    flips = {flipped for _, flipped in transforms}
    assert len(flips) == 1, f"flip differs between frames of a clip: {transforms}"
    assert np.ptp(angles) <= atol, f"rotation differs between frames: {angles}"


def test_augmentations_consistent_within_clip_sleap(two_flies):
    """Every frame of a SLEAP clip must receive the same geometric augmentation.

    Args:
        two_flies: flies fixture used for testing
    """
    nodes = ["head", "thorax", "abdomen", "wingL", "wingR", "eyeL", "eyeR"]

    def make_ds(augmentations):
        return SleapDataset(
            slp_files=[two_flies[0]],
            video_files=[two_flies[1]],
            data_dirs=["./data/sleap"],
            crop_size=64,
            chunk=True,
            clip_length=8,
            anchors=nodes,
            augmentations=augmentations,
            seed=0,
        )

    def anchor_points(frame):
        return {
            (int(inst.gt_track_id), node): np.asarray(xy, dtype=float)
            for inst in frame.instances
            for node, xy in inst.centroid.items()
        }

    raw_ds = make_ds(None)
    aug_ds = make_ds({"Rotate": {"limit": 45, "p": 1.0}, "HorizontalFlip": {"p": 0.5}})

    for clip_idx in range(3):
        raw_frames, aug_frames = raw_ds[clip_idx], aug_ds[clip_idx]
        transforms = _clip_transforms(raw_frames, aug_frames, anchor_points)
        _assert_same_transform(transforms, atol=1e-3)

        # Instance.pose must be in the same (augmented) coordinates as the anchors
        for frame in aug_frames:
            for inst in frame.instances:
                for node, xy in inst.centroid.items():
                    np.testing.assert_allclose(inst.pose[node], xy, atol=1e-4)


def test_augmentations_consistent_within_clip_microscopy(ten_icy_particles):
    """Every frame of a microscopy clip must receive the same augmentation.

    Args:
        ten_icy_particles: icy fixture used for testing
    """

    def make_ds(augmentations):
        return MicroscopyDataset(
            videos=[ten_icy_particles[0]],
            tracks=[ten_icy_particles[1]],
            source="icy",
            crop_size=16,
            chunk=True,
            clip_length=8,
            augmentations=augmentations,
            seed=0,
        )

    raw_ds = make_ds(None)
    aug_ds = make_ds({"Rotate": {"limit": 45, "p": 1.0}, "HorizontalFlip": {"p": 0.5}})

    for clip_idx in range(2):
        transforms = _clip_transforms(raw_ds[clip_idx], aug_ds[clip_idx], _bbox_centers)
        # bboxes are built from integer-rounded centroids, so allow a little slack
        _assert_same_transform(transforms, atol=2.0)


def test_augmentations_cell_tracking(cell_tracking):
    """CTC centroids/bboxes/masks must follow the augmented image, per clip.

    Args:
        cell_tracking: HL60 nuclei fixture used for testing
    """
    raw_img_list, gt_list, ctc_track_meta, data_dir = cell_tracking

    def make_ds(augmentations):
        return CellTrackingDataset(
            raw_img_list=raw_img_list,
            gt_list=gt_list,
            data_dirs=data_dir,
            crop_size=32,
            chunk=True,
            clip_length=8,
            ctc_track_meta=ctc_track_meta,
            apply_mask_to_crop=True,
            augmentations=augmentations,
            seed=0,
        )

    def centroids(frame):
        return {
            int(inst.gt_track_id): np.asarray(inst.centroid["centroid"], dtype=float)
            for inst in frame.instances
        }

    raw_frames = make_ds(None)[0]

    # deterministic flip: centroids, bboxes and masks must all be flipped
    flipped_frames = make_ds({"HorizontalFlip": {"p": 1.0}})[0]
    for raw_frame, aug_frame in zip(raw_frames, flipped_frames):
        width = aug_frame.img_shape[-1]
        raw_c, aug_c = centroids(raw_frame), centroids(aug_frame)
        assert raw_c.keys() == aug_c.keys()
        for track_id, (x, y) in raw_c.items():
            np.testing.assert_allclose(aug_c[track_id], [width - 1 - x, y], atol=1e-3)
        for inst in aug_frame.instances:
            ys, xs = torch.nonzero(inst.mask, as_tuple=True)
            np.testing.assert_allclose(
                [xs.float().mean(), ys.float().mean()],
                inst.centroid["centroid"],
                atol=1e-3,
            )

    # random rotation: same (non-trivial) rotation for every frame of the clip
    rotated_frames = make_ds({"Rotate": {"limit": (30, 60), "p": 1.0}})[0]
    transforms = _clip_transforms(raw_frames, rotated_frames, centroids)
    _assert_same_transform(transforms, atol=3.0)
    assert all(25 <= abs(angle) <= 65 for angle, _ in transforms)


def test_clip_augmentation_randomness():
    """Sampled params must be independent across transforms and follow np.random."""
    clip_augmentations = ClipAugmentation(
        build_augmentations(
            {"Rotate": {"limit": 45, "p": 1.0}, "HorizontalFlip": {"p": 0.5}}
        )
    )
    img = np.zeros((16, 16, 3), dtype=np.uint8)

    def sample():
        _, replay = clip_augmentations(image=img, keypoints=[])
        rotate, flip = replay["transforms"]
        matrix = np.array(rotate["params"]["matrix"])
        return np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])), flip["applied"]

    # reproducible from numpy's global seed (which torch reseeds per worker)
    np.random.seed(123)
    first = [sample() for _ in range(5)]
    np.random.seed(123)
    assert [sample() for _ in range(5)] == first

    # the draws of different transforms must not be correlated
    samples = np.array([sample() for _ in range(500)], dtype=float)
    assert abs(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]) < 0.2


def test_augmentations_differ_across_workers_and_epochs(cell_tracking):
    """Forked DataLoader workers must not replay identical augmentations.

    Args:
        cell_tracking: HL60 nuclei fixture used for testing
    """
    raw_img_list, gt_list, ctc_track_meta, data_dir = cell_tracking
    kwargs = dict(
        raw_img_list=raw_img_list,
        gt_list=gt_list,
        data_dirs=data_dir,
        crop_size=32,
        chunk=True,
        clip_length=2,
        ctc_track_meta=ctc_track_meta,
    )
    raw_ds = CellTrackingDataset(**kwargs)
    aug_ds = CellTrackingDataset(
        **kwargs, augmentations={"Rotate": {"limit": 45, "p": 1.0}}
    )
    loader = DataLoader(
        aug_ds, batch_size=1, num_workers=2, collate_fn=aug_ds.no_batching_fn
    )

    def centroids(frame):
        return {
            int(inst.gt_track_id): np.asarray(inst.centroid["centroid"], dtype=float)
            for inst in frame.instances
        }

    def first_frame_angles():
        angles = []
        for clip_idx, batch in enumerate(loader):
            ((angle, _),) = _clip_transforms(
                raw_ds[clip_idx][:1], batch[0][:1], centroids
            )
            angles.append(angle)
        return np.array(angles)

    epoch_0, epoch_1 = first_frame_angles(), first_frame_angles()
    # with 2 workers, clips 0 and 1 are the first clips of different workers
    assert abs(epoch_0[0] - epoch_0[1]) > 1
    assert np.abs(epoch_0 - epoch_1).max() > 1


# if __name__ == "__main__":
#     from tests.fixtures.datasets import cell_tracking
#     cell_tracking_args = cell_tracking(("/root/vast/mustafa/dreem-experiments/src/dreem/tests/data/microscopy/cell_tracking"))
#     test_cell_tracking_dataset(cell_tracking_args)
