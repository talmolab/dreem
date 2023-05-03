from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.datasets.microscopy_dataset import MicroscopyDataset


def test_sleap_dataset(two_flies):
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
