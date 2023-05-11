"""Test training logic."""
import torch
from biogtr.training.losses import AssoLoss
from biogtr.models.gtr_runner import GTRRunner
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer

# todo: add named tensor tests
# todo: add fixtures

torch.set_default_device("cpu")


def test_asso_loss():
    """Test asso loss."""
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "gt_track_ids": torch.arange(num_detected),
                "bboxes": torch.rand(size=(num_detected, 4)),
            }
        )

    asso_loss = AssoLoss(neg_unmatched=True, asso_weight=10.0)

    asso_preds = torch.rand(size=(1, 100, 100))

    loss = asso_loss(asso_preds, instances)

    assert len(loss.size()) == 0
    assert type(loss.item()) == float


def test_gtr_runner():
    """Test GTR Runner."""
    feats = 512
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)
    n_batches = 2
    instances = []
    train_ds = []
    epochs = 3
    for i in range(n_batches):
        for j in range(num_frames):
            instances.append(
                {
                    "video_id": torch.tensor(0),
                    "frame_id": torch.tensor(j),
                    "img_shape": torch.tensor(img_shape),
                    "num_detected": torch.tensor([num_detected]),
                    "crops": torch.rand(size=(num_detected, 1, 64, 64)),
                    "bboxes": torch.rand(size=(num_detected, 4)),
                    "gt_track_ids": torch.arange(num_detected),
                    "pred_track_ids": torch.tensor([-1] * num_detected),
                }
            )
        train_ds.append([instances])

    embedding_meta = {
        "embedding_type": "fixed_pos",
        "kwargs": {"temperature": num_detected, "scale": num_frames, "normalize": True},
    }

    tracking_transformer = GlobalTrackingTransformer(
        encoder_model="resnet18",
        encoder_cfg={"weights": "ResNet18_Weights.DEFAULT"},
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
        embedding_meta=embedding_meta,
        return_embedding=True,
    )
    loss = AssoLoss()
    tracker_cfg = {
        "window_size": 8,
        "use_vis_feats": True,
        "overlap_thresh": 0.01,
        "mult_thresh": True,
        "decay_time": None,
        "iou": None,
        "max_center_dist": None,
    }
    gtr_runner = GTRRunner(tracking_transformer, tracker_cfg, loss)

    for epoch in range(epochs):
        for i, batch in enumerate(train_ds):
            assert gtr_runner.model.training
            metrics = gtr_runner.training_step(batch, i)
            assert "loss" in metrics and "sw_cnt" not in metrics
            assert metrics["loss"].requires_grad
        for j, batch in enumerate(train_ds):
            gtr_runner.eval()
            with torch.no_grad():
                metrics = gtr_runner.validation_step(batch, j)
            assert "loss" in metrics and "sw_cnt" in metrics
            assert not metrics["loss"].requires_grad
        gtr_runner.train()
    for k, batch in enumerate(train_ds):
        gtr_runner.eval()
        with torch.no_grad():
            metrics = gtr_runner.test_step(batch, k)
        assert "loss" in metrics and "sw_cnt" in metrics
        assert not metrics["loss"].requires_grad
