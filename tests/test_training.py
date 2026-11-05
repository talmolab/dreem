"""Test training logic."""

import os

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from dreem.io import Config, Frame, Instance
from dreem.models import GTRRunner
from dreem.training.losses import AssoLoss
from dreem.training.train import run

# TODO: add named tensor tests
# TODO: use temp dir and cleanup after tests (https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html)


def test_asso_loss():
    """Test asso loss."""
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(Instance(gt_track_id=j, bbox=torch.rand(size=(1, 4))))
        frames.append(
            Frame(video_id=0, frame_id=i, instances=instances, img_shape=img_shape)
        )

    asso_loss = AssoLoss(neg_unmatched=True, asso_weight=10.0)

    asso_preds = torch.rand(size=(1, 100, 100))

    loss = asso_loss(asso_preds, frames)

    assert len(loss.size()) == 0
    assert isinstance(loss.item(), float)


def _asso_gt_loop(pred_box, pred_time, target_box, target_time, target_inst_id, n_t):
    """The per-(instance, frame) loop AssoLoss._get_asso_gt replaced, for comparison."""
    import torchvision

    ious = torchvision.ops.box_iou(pred_box, target_box)
    ious[pred_time[:, None] != target_time[None, :]] = -1.0
    inst_ids = torch.unique(target_inst_id[target_inst_id > -1])
    K, N, T = len(inst_ids), len(pred_box), len(n_t)
    match_cues = pred_box.new_full((N,), -1, dtype=torch.long)
    asso_gt = pred_box.new_zeros((K, T), dtype=torch.long)
    ious_per_frame = ious.split(n_t, dim=0)
    for k, inst_id in enumerate(inst_ids):
        target_inds = target_inst_id == inst_id
        base_ind = 0
        for t in range(T):
            iou_t = ious_per_frame[t][:, target_inds]
            if iou_t.numel() == 0:
                asso_gt[k, t] = n_t[t]
            else:
                val, inds = iou_t.max(dim=0)
                ind = inds[val > 0.0]
                assert len(ind) <= 1
                if len(ind) == 1:
                    asso_gt[k, t] = ind[0].item()
                    match_cues[base_ind + ind[0].item()] = k
                else:
                    asso_gt[k, t] = n_t[t]
            base_ind += n_t[t]
    return asso_gt, match_cues


@pytest.mark.parametrize("seed", range(5))
def test_asso_gt_matches_loop(seed):
    """The vectorized association ground truth equals the per-pair loop's."""
    gen = torch.Generator().manual_seed(seed)
    n_t = torch.randint(0, 6, (7,), generator=gen).tolist()
    n_t[0] = max(n_t[0], 1)
    ids, times, boxes = [], [], []
    for t, n in enumerate(n_t):
        # each track at most once per frame; some instances untracked (-1)
        ids += torch.randperm(8, generator=gen)[:n].tolist()
        times += [t] * n
        xy = torch.rand(n, 2, generator=gen) * 60
        boxes.append(
            torch.cat([xy, xy + 10 + torch.rand(n, 2, generator=gen) * 20], dim=1)
        )
    ids = torch.tensor(ids)
    ids[torch.rand(len(ids), generator=gen) < 0.2] = -1
    box, time = torch.cat(boxes), torch.tensor(times)
    # some boxes overlap a neighbour; one degenerate (zero-area) box
    box[1] = box[0] + 2.0
    box[-1, 2:] = box[-1, :2]

    loss = AssoLoss(neg_unmatched=True, asso_weight=10.0)
    asso_gt, match_cues = loss._get_asso_gt(box, time, box, time, ids, n_t)
    ref_gt, ref_cues = _asso_gt_loop(box, time, box, time, ids, n_t)

    assert torch.equal(asso_gt, ref_gt)
    assert torch.equal(match_cues, ref_cues)


def test_asso_gt_without_tracked_instances():
    """No tracked gt instance gives an empty ground truth and no matches."""
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    time = torch.tensor([0, 1])
    loss = AssoLoss(neg_unmatched=True, asso_weight=10.0)
    asso_gt, match_cues = loss._get_asso_gt(
        box, time, box, time, torch.tensor([-1, -1]), [1, 1]
    )

    assert asso_gt.shape == (0, 2)
    assert match_cues.tolist() == [-1, -1]


def test_basic_gtr_runner():
    """Test basic GTR Runner."""
    num_frames = 2
    num_detected = 3
    img_shape = (1, 128, 128)
    n_batches = 2
    train_ds = []
    epochs = 1
    frame_ind = 0
    for i in range(n_batches):
        frames = []
        for j in range(num_frames):
            instances = []
            for k in range(num_detected):
                instances.append(
                    Instance(
                        gt_track_id=k,
                        pred_track_id=-1,
                        bbox=torch.rand(size=(1, 1, 4)),
                        crop=torch.randn(size=img_shape),
                    ),
                )

            frames.append(
                Frame(
                    video_id=0,
                    frame_id=frame_ind,
                    instances=instances,
                    img_shape=img_shape,
                )
            )
            frame_ind += 1
        train_ds.append(frames)
    gtr_runner = GTRRunner()

    optim_scheduler = gtr_runner.configure_optimizers()

    assert isinstance(optim_scheduler["optimizer"], torch.optim.Adam)
    assert isinstance(
        optim_scheduler["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    optim_cfg = {"name": "SGD", "lr": 1e-3}
    scheduler_cfg = {"name": "CosineAnnealingLR", "T_max": 1}

    gtr_runner = GTRRunner(optimizer_cfg=optim_cfg, scheduler_cfg=scheduler_cfg)
    optim_scheduler = gtr_runner.configure_optimizers()

    assert isinstance(optim_scheduler["optimizer"], torch.optim.SGD)
    assert isinstance(
        optim_scheduler["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.CosineAnnealingLR,
    )

    for epoch in range(epochs):
        for i, batch in enumerate(train_ds):
            gtr_runner.train()
            assert gtr_runner.model.training
            metrics = gtr_runner.training_step([batch], i)
            assert "loss" in metrics
            assert metrics["loss"].requires_grad

        for j, batch in enumerate(train_ds):
            gtr_runner.eval()
            with torch.no_grad():
                metrics = gtr_runner.validation_step([batch], j)
            assert "loss" in metrics
            assert not metrics["loss"].requires_grad

    for k, batch in enumerate(train_ds):
        gtr_runner.eval()
        with torch.no_grad():
            metrics = gtr_runner.test_step([batch], k)


# temp fix for actions test, still need to debug
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Silent fail on GitHub Actions",
)
def test_config_gtr_runner(tmp_path, base_config, params_config, two_flies):
    """Test config GTR Runner."""
    base_cfg = OmegaConf.load(base_config)
    base_cfg["params_config"] = params_config
    cfg = Config(base_cfg)

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    hparams = {
        "dataset.clip_length": 8,
        "trainer.min_epochs": 1,
        "checkpointing.dirpath": model_dir,
        "logging.save_dir": logs_dir,
    }

    cfg.set_hparams(hparams)
    with torch.autograd.set_detect_anomaly(True):
        run(cfg.cfg)


def test_checkpoint_legacy_embeddings(tmp_path):
    """Test that checkpoints record the embedding behavior they were trained with.

    Checkpoints saved before the embedding fixes don't record it and must load in
    legacy mode so they run as they were trained.
    """

    class EmptyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return None

    model_cfg = {
        "encoder_cfg": {"model_name": "resnet18", "pretrained": False, "in_chans": 3},
        "d_model": 64,
        "nhead": 1,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "embedding_meta": {
            "pos": {"mode": "learned", "emb_num": 16},
            "temp": {"mode": "learned", "emb_num": 16},
        },
    }
    model = GTRRunner(model_cfg=model_cfg)
    assert not model.model.legacy_embeddings

    ckpt_path = tmp_path / "model.ckpt"
    trainer = Trainer(
        max_steps=1, accelerator="cpu", logger=False, enable_checkpointing=False
    )
    trainer.fit(model, torch.utils.data.DataLoader(EmptyDataset()))
    trainer.save_checkpoint(ckpt_path)

    checkpoint = torch.load(ckpt_path, weights_only=False)
    assert checkpoint[GTRRunner.LEGACY_EMBEDDINGS_KEY] is False
    assert not GTRRunner.load_from_checkpoint(ckpt_path).model.legacy_embeddings

    # a checkpoint from before the fix has no record of it
    del checkpoint[GTRRunner.LEGACY_EMBEDDINGS_KEY]
    old_ckpt_path = tmp_path / "old_model.ckpt"
    torch.save(checkpoint, old_ckpt_path)
    old_model = GTRRunner.load_from_checkpoint(old_ckpt_path).model
    assert old_model.legacy_embeddings
    assert old_model.transformer.pos_emb.legacy
    assert old_model.transformer.temp_emb.legacy

    # and keeps it when saved again, e.g. after fine-tuning
    resaved = {}
    GTRRunner.load_from_checkpoint(old_ckpt_path).on_save_checkpoint(resaved)
    assert resaved[GTRRunner.LEGACY_EMBEDDINGS_KEY] is True


def test_checkpoint_normalize_image():
    """Test that checkpoints record the image scale they were trained on."""
    model = GTRRunner(
        model_cfg={
            "encoder_cfg": {
                "model_name": "resnet18",
                "pretrained": False,
                "in_chans": 3,
            },
            "d_model": 64,
            "nhead": 1,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
        }
    )

    saved = {}
    model.on_save_checkpoint(saved)
    assert saved[GTRRunner.NORMALIZE_IMAGE_KEY] is True

    # e.g. fine-tuned on [0, 255] CTC crops
    model.normalize_image = False
    model.on_save_checkpoint(saved)
    assert saved[GTRRunner.NORMALIZE_IMAGE_KEY] is False

    model.on_load_checkpoint(saved)
    assert model.normalize_image is False

    # checkpoints from before the scale was recorded leave it unknown
    model.on_load_checkpoint({})
    assert model.normalize_image is None


def test_configure_optimizers_geom_group():
    """RoPE frequencies get their own group: no weight decay and `geom_lr`."""
    model_cfg = {
        "encoder_cfg": {"model_name": "resnet18", "pretrained": False, "in_chans": 1},
        "d_model": 64,
        "nhead": 1,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "embedding_meta": {"embedding_agg_method": "concatenate", "rope": {}},
    }
    optim_cfg = {"name": "Adam", "lr": 1e-4, "weight_decay": 0.01, "geom_lr": 1e-3}
    runner = GTRRunner(model_cfg=model_cfg, optimizer_cfg=optim_cfg)
    optimizer = runner.configure_optimizers()["optimizer"]

    rest, geom = optimizer.param_groups
    omega = runner.model.transformer.attn_head.rope.omega
    assert [id(p) for p in geom["params"]] == [id(omega)]
    assert geom["weight_decay"] == 0.0 and geom["lr"] == 1e-3
    assert rest["weight_decay"] == 0.01 and rest["lr"] == 1e-4
    assert sum(p.numel() for g in optimizer.param_groups for p in g["params"]) == sum(
        p.numel() for p in runner.parameters()
    )
    assert "geom_lr" in runner.optimizer_cfg  # the runner's own config is untouched
