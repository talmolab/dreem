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
