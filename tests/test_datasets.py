from biogtr.datasets.sleap_dataset import SleapDataset


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
    assert len(instances[0]["gt_track_ids"]) == instances[0]["num_detected"].item()
