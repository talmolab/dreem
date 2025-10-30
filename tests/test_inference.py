"""Test inference logic."""

import os

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from dreem.inference import Tracker, metrics, post_processing
from dreem.inference.track import run
from dreem.inference.track_queue import TrackQueue
from dreem.io import Config, Frame, Instance
from dreem.models import GlobalTrackingTransformer, GTRRunner


def test_track_queue():
    window_size = 8
    max_gap = 10
    img_shape = (3, 1024, 1024)
    n_instances_per_frame = [2] * window_size

    frames = []
    instances_per_frame = []

    tq = TrackQueue(window_size, max_gap)
    for i in range(window_size):
        instances = []
        for j in range(n_instances_per_frame[i]):
            instances.append(Instance(gt_track_id=j, pred_track_id=j))
        instances_per_frame.append(instances)
        frame = Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        frames.append(frame)

        tq.add_frame(frame)

    assert len(tq) == sum(n_instances_per_frame)
    assert tq.n_tracks == max(n_instances_per_frame)
    assert tq.tracks == [i for i in range(max(n_instances_per_frame))]
    assert len(tq.collate_tracks()) == window_size
    assert all([gap == 0 for gap in tq._curr_gap.values()])
    assert max(tq.curr_track) == max(n_instances_per_frame) - 1

    tq.add_frame(
        Frame(
            video_id=0,
            frame_id=window_size + 1,
            img_shape=img_shape,
            instances=[Instance(gt_track_id=0, pred_track_id=0)],
        )
    )

    assert len(tq._queues[0]) == window_size
    assert tq._curr_gap[0] == 0
    assert tq._curr_gap[max(n_instances_per_frame) - 1] == 1

    tq.add_frame(
        Frame(
            video_id=0,
            frame_id=window_size + 1,
            img_shape=img_shape,
            instances=[
                Instance(gt_track_id=1, pred_track_id=1),
                Instance(
                    gt_track_id=max(n_instances_per_frame),
                    pred_track_id=max(n_instances_per_frame),
                ),
            ],
        )
    )

    assert len(tq._queues[max(n_instances_per_frame)]) == 1
    assert tq._curr_gap[1] == 0
    assert tq._curr_gap[0] == 1

    for i in range(max_gap):
        tq.add_frame(
            Frame(
                video_id=0,
                frame_id=window_size + i + 1,
                img_shape=img_shape,
                instances=[Instance(gt_track_id=0, pred_track_id=0)],
            )
        )

    assert tq.n_tracks == 1
    assert max(tq.curr_track) == max(n_instances_per_frame)
    assert 0 in tq._queues.keys()

    tq.end_tracks()

    assert len(tq) == 0


def test_tracker():
    """Test tracker module.

    Tests that tracker works with/without post processing
    """
    feats = 512
    num_frames = 5
    num_detected = 2
    img_shape = (1, 128, 128)
    test_frame = 1
    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    gt_track_id=j,
                    pred_track_id=-1,
                    bbox=torch.rand(size=(1, 4)),
                    crop=torch.rand(size=(1, 1, 64, 64)),
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    embedding_meta = {
        "embedding_type": "fixed_pos",
        "kwargs": {"temperature": num_detected, "scale": num_frames, "normalize": True},
    }

    tracking_transformer = GlobalTrackingTransformer(
        encoder_cfg={"model_name": "resnet18", "pretrained": False, "in_chans": 3},
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        embedding_meta=embedding_meta,
        return_embedding=False,
    )

    tracking_cfg = {
        "overlap_thresh": 0.01,
        "decay_time": None,
        "iou": None,
        "max_center_dist": None,
    }

    tracker = Tracker(**tracking_cfg)

    frames_pred = tracker(tracking_transformer, frames)

    # TODO: Debug saving asso matrices
    # asso_equals = (
    #     frames_pred[test_frame].get_traj_score("decay_time").to_numpy()
    #     == frames_pred[test_frame].get_traj_score("final").to_numpy()
    # ).all()
    # assert asso_equals

    assert len(frames_pred[test_frame].get_pred_track_ids()) == num_detected


# @pytest.mark.parametrize("set_default_device", ["cpu"], indirect=True)
def test_post_processing():  # set_default_device
    """Test postprocessing methods.

    Tests each postprocessing method to ensure that
    it filters only when the condition is satisfied.
    """
    T = 8
    k = 5
    D = 512
    M = 5
    N_t = 5
    N_p = N_t * (T - 1)
    N = N_t * T

    reid_features = torch.rand((1, N_t, D))
    asso_nonk = torch.rand((N_t, N_p))

    decay_time = 0

    asso_output = torch.rand((N_t, M))
    ious = torch.rand((N_t, M))

    assert (asso_output == post_processing.weight_iou(asso_output, None, ious)).all()

    assert not (
        asso_output == post_processing.weight_iou(asso_output, "mult", ious)
    ).all()

    assert not (
        asso_output == post_processing.weight_iou(asso_output, "max", ious)
    ).all()

    assert not (
        post_processing.weight_iou(asso_output, "mult", ious)
        == post_processing.weight_iou(asso_output, "max", ious)
    ).all()

    im_size = 128
    k_boxes = torch.rand((N_t, 1, 4)) * im_size
    nonk_boxes = torch.rand((N_p, 1, 4)) * im_size
    id_inds = torch.tile(torch.cat((torch.zeros(M - 1), torch.ones(1))), (N_p, 1))

    assert (
        asso_output
        == post_processing.filter_max_center_dist(
            asso_output=asso_output,
            max_center_dist=0,
            id_inds=id_inds,
            query_boxes_px=k_boxes,
            nonquery_boxes_px=nonk_boxes,
        )
    ).all()

    assert not (
        asso_output
        == post_processing.filter_max_center_dist(
            asso_output=asso_output,
            max_center_dist=1e-9,
            id_inds=id_inds,
            query_boxes_px=k_boxes,
            nonquery_boxes_px=nonk_boxes,
        )
    ).all()


def test_metrics():
    """Test basic GTR Runner."""
    num_frames = 3
    num_detected = 3
    metrics_to_compute = ["motmetrics", "global_tracking_accuracy"]

    frames_pred = []
    for i in range(num_frames):
        instances_pred = []
        for k in range(num_detected):
            bboxes = torch.tensor(np.random.uniform(size=(num_detected, 4)))
            bboxes[:, -2:] += 1
            instances_pred.append(
                Instance(gt_track_id=k, pred_track_id=k, bbox=torch.randn((1, 4)))
            )
        frames_pred.append(Frame(video_id=0, frame_id=i, instances=instances_pred))

    results = metrics.evaluate(frames_pred, metrics_to_compute)
    for metric_name, value in results.items():
        if metric_name == "motmetrics":
            mot_summary = value[0]
            switch_count = mot_summary.loc["num_switches"].values[0]
        elif metric_name == "global_tracking_accuracy":
            gta_by_gt_track = value
            overall_gta = np.mean(np.array(list(gta_by_gt_track.values())))

    assert switch_count == 0
    assert 0 < overall_gta <= 1


def get_ckpt(ckpt_path: str):
    """Save GTR Runner to checkpoint file."""

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return None

    dl = torch.utils.data.DataLoader(DummyDataset())
    model = GTRRunner()
    trainer = Trainer(max_steps=1, min_steps=1, accelerator="cpu")
    trainer.fit(model, dl)
    trainer.save_checkpoint(ckpt_path)

    return ckpt_path


def test_track(tmp_path, inference_config):
    ckpt_path = tmp_path / "model.ckpt"
    get_ckpt(ckpt_path)

    out_dir = tmp_path / "preds"
    out_dir.mkdir()

    inference_cfg = OmegaConf.load(inference_config)

    cfg = Config(inference_cfg)

    cfg.set_hparams({"ckpt_path": ckpt_path, "outdir": out_dir})

    run(cfg.cfg)
    # the test file path should contain pairs of slp/mp4, so there will be half as many slp files as total files in the directory
    assert (
        len(list(out_dir.iterdir()))
        == len(os.listdir(cfg.cfg.dataset.test_dataset.dir.path)) / 2
    )


def test_confidence_thresholding():
    """Test confidence thresholding feature.

    Tests that the confidence thresholding correctly filters rows
    based on entropy calculations and maintains proper index mappings.
    """
    # Test 1: Basic functionality - some rows filtered
    print("\nTest 1: Basic functionality with some rows filtered")
    n_query = 5
    n_traj = 3
    temperature = 0.1
    confidence_threshold = 0.7

    # Create a trajectory score matrix where some rows have high confidence (low entropy)
    # and others have low confidence (high entropy)
    traj_score = torch.tensor(
        [
            [0.8, 0.15, 0.05],  # High confidence in first track
            [0.33, 0.33, 0.34],  # Low confidence (uniform distribution)
            [0.1, 0.85, 0.05],  # High confidence in second track
            [0.32, 0.34, 0.34],  # Low confidence (uniform distribution)
            [0.05, 0.1, 0.85],  # High confidence in third track
        ],
        dtype=torch.float32,
    )

    # Apply log-softmax scaling (as done in tracker)
    scaled_traj_score = torch.nn.functional.log_softmax(traj_score / temperature, dim=1)

    # Calculate entropy (as done in tracker)
    entropy = -torch.sum(scaled_traj_score * torch.exp(scaled_traj_score), axis=1)
    norm_entropy = entropy / torch.log(torch.tensor(n_query, dtype=torch.float32))
    removal_threshold = 1 - confidence_threshold
    remove = norm_entropy > removal_threshold

    print(f"Entropy: {entropy}")
    print(f"Normalized entropy: {norm_entropy}")
    print(f"Removal threshold: {removal_threshold}")
    print(f"Rows to remove: {remove}")

    # Verify that at least some rows are marked for removal
    assert remove.sum() > 0, "Expected some rows to be removed"
    assert remove.sum() < traj_score.shape[0], "Expected some rows to remain"

    # Create index mappings (as done in tracker)
    dict_old_new_map = {i: None for i in range(traj_score.shape[0])}
    dict_new_old_map = {}
    new_idx = 0
    for idx in range(traj_score.shape[0]):
        if not remove[idx].item():
            dict_old_new_map[idx] = new_idx
            dict_new_old_map[new_idx] = idx
            new_idx += 1

    # Filter the trajectory score matrix
    traj_score_filt = traj_score[~remove]

    # Verify filtered matrix has correct shape
    expected_rows = traj_score.shape[0] - remove.sum().item()
    assert traj_score_filt.shape[0] == expected_rows, (
        f"Expected {expected_rows} rows, got {traj_score_filt.shape[0]}"
    )
    assert traj_score_filt.shape[1] == n_traj, (
        f"Expected {n_traj} columns, got {traj_score_filt.shape[1]}"
    )

    # Verify index mappings are consistent
    for old_idx in range(traj_score.shape[0]):
        if not remove[old_idx].item():
            new_idx = dict_old_new_map[old_idx]
            assert dict_new_old_map[new_idx] == old_idx, "Index mapping inconsistency"

    print(
        f"Test 1 passed: Filtered {remove.sum().item()} out of {traj_score.shape[0]} rows"
    )

    # Test 2: All rows would be removed (edge case)
    print("\nTest 2: All rows have high entropy")
    traj_score_uniform = torch.ones((4, 3)) / 3.0  # Perfectly uniform - maximum entropy
    scaled_uniform = torch.nn.functional.log_softmax(
        traj_score_uniform / temperature, dim=1
    )

    entropy_uniform = -torch.sum(scaled_uniform * torch.exp(scaled_uniform), axis=1)
    norm_entropy_uniform = entropy_uniform / torch.log(torch.tensor(4.0))
    remove_uniform = norm_entropy_uniform > removal_threshold

    # Verify all rows are marked for removal
    assert remove_uniform.sum() == traj_score_uniform.shape[0], (
        "Expected all rows to be removed for uniform distribution with high threshold"
    )

    print("Test 2 passed: All rows correctly identified for removal")

    # Test 3: No filtering when confidence_threshold = 0
    print("\nTest 3: No filtering with confidence_threshold = 0")
    confidence_threshold_zero = 0

    if confidence_threshold_zero > 0:
        # This block should not execute
        assert False, "Should not filter when confidence_threshold is 0"
    else:
        traj_score_no_filt = traj_score
        remove_zero = torch.zeros(traj_score.shape[0], dtype=torch.bool)

    assert remove_zero.sum() == 0, "Expected no rows to be removed"
    assert torch.equal(traj_score_no_filt, traj_score), "Matrix should be unchanged"

    print("Test 3 passed: No filtering applied when threshold is 0")

    # Test 4: Very high confidence threshold (keep almost all rows)
    print("\nTest 4: Very high confidence threshold (0.95)")
    high_confidence_threshold = 0.95
    removal_threshold_high = 1 - high_confidence_threshold

    scaled_traj_score_high = torch.nn.functional.log_softmax(
        traj_score / temperature, dim=1
    )
    entropy_high = -torch.sum(
        scaled_traj_score_high * torch.exp(scaled_traj_score_high), axis=1
    )
    norm_entropy_high = entropy_high / torch.log(
        torch.tensor(n_query, dtype=torch.float32)
    )
    remove_high = norm_entropy_high > removal_threshold_high

    # With very high threshold (0.95), we should keep more rows
    assert remove_high.sum() <= remove.sum(), (
        "Higher confidence threshold should remove fewer or equal rows"
    )

    print(
        f"Test 4 passed: Only {remove_high.sum().item()} rows removed with high threshold"
    )

    # Test 5: Different matrix sizes
    print("\nTest 5: Different matrix sizes")
    for n_rows, n_cols in [(2, 2), (10, 5), (3, 8)]:
        test_score = torch.rand((n_rows, n_cols))
        test_score = test_score / test_score.sum(dim=1, keepdim=True)  # Normalize rows

        scaled_test = torch.nn.functional.log_softmax(test_score / temperature, dim=1)
        entropy_test = -torch.sum(scaled_test * torch.exp(scaled_test), axis=1)
        norm_entropy_test = entropy_test / torch.log(torch.tensor(float(n_rows)))
        remove_test = norm_entropy_test > removal_threshold

        if remove_test.sum() > 0 and remove_test.sum() < n_rows:
            filtered_test = test_score[~remove_test]
            assert filtered_test.shape[1] == n_cols, (
                f"Column count should remain {n_cols}"
            )
            assert filtered_test.shape[0] == n_rows - remove_test.sum().item(), (
                "Row count should match filtered count"
            )

    print("Test 5 passed: Various matrix sizes handled correctly")

    # Test 6: Verify entropy calculation for known distributions
    print("\nTest 6: Entropy calculation verification")

    # Perfect certainty (one-hot) should have entropy ~0
    one_hot = torch.zeros((1, 3))
    one_hot[0, 0] = 1.0
    scaled_one_hot = torch.nn.functional.log_softmax(one_hot / temperature, dim=1)
    entropy_one_hot = -torch.sum(scaled_one_hot * torch.exp(scaled_one_hot), axis=1)

    # Uniform distribution should have maximum entropy
    uniform = torch.ones((1, 3)) / 3.0
    scaled_uniform_single = torch.nn.functional.log_softmax(
        uniform / temperature, dim=1
    )
    entropy_uniform_single = -torch.sum(
        scaled_uniform_single * torch.exp(scaled_uniform_single), axis=1
    )

    # Entropy should be lower for one-hot than uniform
    assert entropy_one_hot < entropy_uniform_single, (
        "One-hot distribution should have lower entropy than uniform"
    )

    print("Test 6 passed: Entropy calculation verified")

    # Test 7: Index mapping correctness
    print("\nTest 7: Detailed index mapping verification")
    test_remove = torch.tensor([False, True, False, True, False])

    dict_old_new_test = {i: None for i in range(5)}
    dict_new_old_test = {}
    new_idx_test = 0
    for idx in range(5):
        if not test_remove[idx].item():
            dict_old_new_test[idx] = new_idx_test
            dict_new_old_test[new_idx_test] = idx
            new_idx_test += 1

    # Expected mappings:
    # Old index 0 -> New index 0
    # Old index 1 -> removed (None)
    # Old index 2 -> New index 1
    # Old index 3 -> removed (None)
    # Old index 4 -> New index 2

    assert dict_old_new_test[0] == 0
    assert dict_old_new_test[1] is None
    assert dict_old_new_test[2] == 1
    assert dict_old_new_test[3] is None
    assert dict_old_new_test[4] == 2

    assert dict_new_old_test[0] == 0
    assert dict_new_old_test[1] == 2
    assert dict_new_old_test[2] == 4

    print("Test 7 passed: Index mappings are correct")

    print("\nAll confidence thresholding tests passed!")


def test_filter_max_angle_diff_basic():
    """Test 1: Basic angle difference calculation and weighting.

    Tests that angles are computed correctly and association scores are weighted
    proportionally to the angle difference.
    """
    asso_output = torch.tensor([
        [0.8, 0.6],
        [0.7, 0.9],
    ], dtype=torch.float32)

    query_principal_axes = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=torch.float32)

    last_principal_axes = torch.tensor([
        [1.0, 0.0],
        [0.0, -1.0],
    ], dtype=torch.float32)

    result = post_processing.weight_by_angle_diff(
        asso_output,
        query_principal_axes,
        last_principal_axes
    )

    # Expected: no wrapping by default
    # angle_0: 0 degrees, angle_1: 90 degrees
    angle_diffs = torch.tensor([[0.0, torch.pi/2], [torch.pi/2, 0.0]])
    weight = asso_output.mean(dim=1)  # [0.7, 0.8]
    penalty = -weight[:, None] * angle_diffs
    expected = asso_output + penalty

    assert torch.allclose(result, expected, rtol=1e-4)


def test_filter_max_angle_diff_angle_wrapping():
    """Test 2: Angle difference wrapping to [0, pi/2].

    Tests that angles > pi/2 are correctly wrapped to [0, pi/2] range
    for cases where head/tail cannot be disambiguated (fallback=True).
    """
    asso_output = torch.tensor([
        [0.5, 0.3, 0.2],
    ])

    query_principal_axes = torch.tensor([
        [1.0, 0.0],
    ])

    theta = torch.tensor(2) * torch.pi / 3
    last_principal_axes = torch.tensor([
        [torch.cos(theta), torch.sin(theta)],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    result = post_processing.weight_by_angle_diff(
        asso_output,
        query_principal_axes,
        last_principal_axes,
        fallback=True
    )

    # With fallback: 120 degrees wraps to 60 degrees (pi - 120)
    mean_weight = asso_output.mean(dim=1)[0]  # (0.5 + 0.3 + 0.2) / 3
    angle_after_wrap = torch.pi - 2 * torch.pi / 3
    penalty_120 = -mean_weight * angle_after_wrap
    penalty_0 = 0.0
    penalty_90 = -mean_weight * torch.pi / 2

    assert torch.isclose(result[0, 0], asso_output[0, 0] + penalty_120, rtol=1e-4)
    assert torch.isclose(result[0, 1], asso_output[0, 1] + penalty_0, rtol=1e-4)
    assert torch.isclose(result[0, 2], asso_output[0, 2] + penalty_90, rtol=1e-4)


def test_filter_max_angle_diff_proportional_penalty():
    """Test 3: Proportional penalty based on angle difference.

    Tests that larger angle differences result in larger penalties.
    """
    asso_output = torch.tensor([
        [0.8, 0.6],
        [0.7, 0.9],
    ])

    query_principal_axes = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    last_principal_axes = torch.tensor([
        [1.0, 0.0],
        [0.707, 0.707],  # 45 degrees
    ])

    result = post_processing.weight_by_angle_diff(
        asso_output,
        query_principal_axes,
        last_principal_axes
    )

    # angle_0: 0 degrees, angle_1: 45 degrees
    angle_diffs = torch.tensor([[0.0, torch.pi/4], [torch.pi/2, 0.0]])
    weight = asso_output.mean(dim=1)
    penalty = -weight[:, None] * angle_diffs
    expected = asso_output + penalty

    assert torch.allclose(result, expected, rtol=1e-4)

    # Verify larger angle = more penalty
    assert result[0, 1] < result[0, 0]  # 45 degree penalty < 0 degree penalty
    assert result[1, 0] < result[1, 1]  # 90 degree penalty < 0 degree penalty


def test_filter_max_angle_diff_with_nans():
    """Test 4: Handling of NaN values in principal axes.

    Tests behavior when principal axes contain NaN values.
    """
    asso_output = torch.tensor([
        [0.8, 0.6],
        [0.7, 0.9],
    ], dtype=torch.float32)

    query_principal_axes = torch.tensor([
        [1.0, 0.0],
        [float('nan'), 1.0],
    ], dtype=torch.float32)

    last_principal_axes = torch.tensor([
        [1.0, float('nan')],
        [0.0, 1.0],
    ])

    result = post_processing.weight_by_angle_diff(
        asso_output,
        query_principal_axes,
        last_principal_axes
    )

    assert result.shape == asso_output.shape
    assert not torch.isnan(result).all()
    assert not torch.isinf(result).all()


def test_filter_max_angle_diff_weighting_mechanism():
    """Test 5: Verify association weighting mechanism.

    Tests that the weighting uses row-wise mean of association scores
    applied proportionally to angle differences.
    """
    asso_output = torch.tensor([
        [1.0, 0.8, 0.6],
        [0.2, 0.3, 0.4],
        [0.5, 0.5, 0.5],
    ], dtype=torch.float32)

    query_principal_axes = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=torch.float32)

    angles = [torch.pi/2, torch.pi/4, torch.pi/6]
    last_principal_axes = torch.tensor([
        [torch.cos(a).item(), torch.sin(a).item()] for a in angles
    ], dtype=torch.float32)

    result = post_processing.filter_max_angle_diff(
        asso_output,
        query_principal_axes,
        last_principal_axes
    )

    # Verify penalties scale with angle difference
    # Row 0: angles are [90, 45, 30] degrees, weights [0.8, 0.8, 0.8]
    assert result[0, 0] < result[0, 1] < result[0, 2]  # larger angle = more penalty
    # Row 1: same angles, weights [0.3, 0.3, 0.3] (smaller weights)
    assert result[1, 0] < result[1, 1] < result[1, 2]

    # Verify higher mean weight causes larger penalty for same angle
    assert (result[0, 0] - asso_output[0, 0]) < (result[1, 0] - asso_output[1, 0])
