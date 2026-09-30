"""Tests for `config.py`."""

import glob
import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from dreem.io import Config
from dreem.io.config import _TRAINING_ONLY_CONFIG_SECTIONS
from dreem.models import GlobalTrackingTransformer, GTRRunner


def test_init(base_config, params_config):
    """Test the `__init__` function of `Config`.

    Load + merge

    Args:
        base_config: the initial config params
        params_config: the config params to override
    """
    base_cfg = OmegaConf.load(base_config)
    base_cfg["params_config"] = params_config
    cfg = Config(base_cfg)

    assert cfg.cfg.model.num_encoder_layers == 2
    assert cfg.cfg.dataset.train_dataset.clip_length == 32
    assert cfg.cfg.dataset.train_dataset.padding == 5
    assert "val_dataset" in cfg.cfg.dataset


def test_setter(base_config):
    """Test the setter function of `Config`.

    set hyperparameters

    Args:
        base_config: the initial config params
    """
    base_cfg = OmegaConf.load(base_config)
    cfg = Config(base_cfg)
    hparams = {
        "model.num_encoder_layers": 1,
        "logging.name": "test_config",
        "dataset.train_dataset.chunk": False,
    }

    assert cfg.set_hparams(hparams)
    assert cfg.cfg.model.num_encoder_layers == 1
    assert cfg.cfg.tracker.window_size == 8

    hparams = {"test_config": -1}

    assert cfg.set_hparams(hparams)
    assert "test_config" in cfg.cfg
    assert cfg.cfg.test_config == -1


def test_getters(base_config, sleap_data_dir):
    """Test each getter function in the config class.

    Args:
        base_config: the config params to override
        sleap_data_dir: path to the sleap test data directory
    """
    base_cfg = OmegaConf.load(base_config)
    cfg = Config(base_cfg)

    model = cfg.get_model()
    assert isinstance(model, GlobalTrackingTransformer)
    assert model.transformer.d_model == 512

    tracker_cfg = cfg.get_tracker_cfg()
    assert set(
        [
            "window_size",
            "use_vis_feats",
            "overlap_thresh",
            "mult_thresh",
            "decay_time",
            "iou",
            "max_center_dist",
        ]
    ) == set(tracker_cfg.keys())
    assert tracker_cfg["window_size"] == 8

    gtr_runner = cfg.get_gtr_runner()
    assert isinstance(gtr_runner, GTRRunner)
    assert gtr_runner.model.transformer.d_model == 512

    ds = cfg.get_dataset("train")
    assert ds.clip_length == 4
    assert len(ds.label_files) == len(ds.vid_files) == 5
    ds = cfg.get_dataset("val")
    assert ds.clip_length == 8

    cfg.set_hparams(
        {
            "dataset.train_dataset.dir": {
                "path": sleap_data_dir,
                "labels_suffix": ".slp",
                "vid_suffix": ".mp4",
            }
        }
    )
    ds = cfg.get_dataset("train")
    assert len(ds.label_files) == len(ds.vid_files) == 5

    optim = cfg.get_optimizer(model.parameters())
    assert isinstance(optim, torch.optim.Adam)

    scheduler = cfg.get_scheduler(optim)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    label_paths, data_path = cfg.get_data_paths(
        "train",
        {
            "dir": {
                "path": sleap_data_dir,
                "labels_suffix": ".slp",
                "vid_suffix": ".mp4",
            }
        },
    )
    assert len(label_paths) == len(data_path) == 5
    for label_path, video_path in zip(label_paths, data_path):
        assert Path(label_path).stem == Path(video_path).stem


def _touch(directory, *names):
    """Create empty files in `directory`."""
    for name in names:
        (directory / name).touch()


def _dir_cfg(path):
    """Dataset config that discovers .slp labels and .mp4 videos in `path`."""
    return {"dir": {"path": str(path), "labels_suffix": ".slp", "vid_suffix": ".mp4"}}


def test_get_data_paths_pairs_by_name(base_config, tmp_path, monkeypatch):
    """Labels and videos found in a directory are paired by name, not listing order.

    Args:
        base_config: the initial config params
        tmp_path: directory for the fake labels and videos
        monkeypatch: used to scramble the directory listing order
    """
    _touch(
        tmp_path,
        "a.slp",
        "b.predictions.proofread.slp",
        "c.mp4.predictions.slp",
        "c_noisy.slp",
        "a.mp4",
        "b.mp4",
        "c.mp4",
        "c_noisy.mp4",
        "unlabeled.mp4",
    )
    # Filesystems list directories in arbitrary order; make it adversarial.
    real_glob = glob.glob
    monkeypatch.setattr(
        glob, "glob", lambda pattern: sorted(real_glob(pattern), reverse=True)
    )

    cfg = Config(OmegaConf.load(base_config))
    labels, videos = cfg.get_data_paths("test", _dir_cfg(tmp_path))

    assert {Path(lab).name: Path(vid).name for lab, vid in zip(labels, videos)} == {
        "a.slp": "a.mp4",
        "b.predictions.proofread.slp": "b.mp4",
        "c.mp4.predictions.slp": "c.mp4",
        "c_noisy.slp": "c_noisy.mp4",
    }


def test_get_data_paths_uses_video_named_in_labels(
    base_config, tmp_path, sleap_data_dir
):
    """A labels file whose name matches no video pairs with the video it references.

    Args:
        base_config: the initial config params
        tmp_path: directory for the labels and fake videos
        sleap_data_dir: directory holding two_flies.slp, which references two_flies.mp4
    """
    shutil.copy(sleap_data_dir / "two_flies.slp", tmp_path / "session_labels.slp")
    _touch(tmp_path, "two_flies.mp4", "other.mp4")

    cfg = Config(OmegaConf.load(base_config))
    labels, videos = cfg.get_data_paths("test", _dir_cfg(tmp_path))

    assert [Path(v).name for v in videos] == ["two_flies.mp4"]


def test_get_data_paths_rejects_unpaired_labels(base_config, tmp_path):
    """A labels file with no video is an error, not a silent mispairing.

    Args:
        base_config: the initial config params
        tmp_path: directory for the fake labels and videos
    """
    _touch(tmp_path, "a.slp", "orphan.slp", "a.mp4")

    cfg = Config(OmegaConf.load(base_config))
    with pytest.raises(ValueError, match="no video found for .*orphan.slp"):
        cfg.get_data_paths("test", _dir_cfg(tmp_path))


def test_get_data_paths_explicit_files(base_config, tmp_path):
    """Explicit `slp_files` / `video_files` lists.

    Args:
        base_config: the initial config params
        tmp_path: directory for the fake labels and videos
    """
    _touch(tmp_path, "a.slp", "b.slp", "a.mp4", "b.mp4")
    a_slp, b_slp = str(tmp_path / "a.slp"), str(tmp_path / "b.slp")
    a_mp4, b_mp4 = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    cfg = Config(OmegaConf.load(base_config))

    # both lists: the user's pairing is kept, even when the names differ
    labels, videos = cfg.get_data_paths(
        "test", {**_dir_cfg(tmp_path), "slp_files": [a_slp], "video_files": [b_mp4]}
    )
    assert (labels, videos) == ([a_slp], [b_mp4])

    with pytest.raises(ValueError, match="same length"):
        cfg.get_data_paths(
            "test",
            {
                **_dir_cfg(tmp_path),
                "slp_files": [a_slp],
                "video_files": [a_mp4, b_mp4],
            },
        )

    # only labels: each is paired with its own video
    labels, videos = cfg.get_data_paths(
        "test", {**_dir_cfg(tmp_path), "slp_files": [b_slp]}
    )
    assert (labels, videos) == ([b_slp], [b_mp4])


def test_missing(base_config):
    """Test cases when keys are missing from config for expected behavior.

    Args:
        base_config: the config params to override
    """
    cfg = Config.from_yaml(base_config)

    key = "model"
    with open_dict(cfg.cfg):
        cfg.cfg.pop(key)
        assert isinstance(cfg.get_model(), GlobalTrackingTransformer)

    cfg = Config.from_yaml(base_config)
    key = "tracker"
    with open_dict(cfg.cfg):
        cfg.cfg.pop(key)
        assert (
            isinstance(cfg.get_tracker_cfg(), dict) and len(cfg.get_tracker_cfg()) == 0
        )

    cfg = Config.from_yaml(base_config)
    keys = ["tracker", "optimizer", "scheduler", "loss", "runner", "model"]
    with open_dict(cfg.cfg):
        for key in keys:
            cfg.cfg.pop(key)
            assert isinstance(cfg.get_gtr_runner(), GTRRunner)


def test_get_trainer_inference_mode():
    """Test that training-only keys are stripped in inference mode."""
    cfg_dict = OmegaConf.create(
        {
            "trainer": {
                "strategy": "ddp_find_unused_parameters_true",
                "max_epochs": 100,
                "accumulate_grad_batches": 4,
                "accelerator": "gpu",
                "devices": 4,
                "enable_progress_bar": True,
            }
        }
    )
    cfg = Config(cfg_dict)

    # Inference mode should strip training-only keys
    trainer = cfg.get_trainer(mode="inference")
    assert trainer.num_devices == 1

    # Verify inference defaults are applied
    cfg2 = Config(OmegaConf.create({"trainer": {"strategy": "ddp"}}))
    trainer2 = cfg2.get_trainer(mode="inference")
    assert trainer2.num_devices == 1

    # Training mode (mode=None) should preserve all keys
    cfg3 = Config(
        OmegaConf.create(
            {
                "trainer": {
                    "max_epochs": 50,
                    "accelerator": "cpu",
                    "devices": 1,
                }
            }
        )
    )
    trainer3 = cfg3.get_trainer(mode=None)
    assert trainer3.max_epochs == 50


def test_get_trainer_inference_defaults():
    """Test that inference defaults are applied when keys are missing."""
    cfg = Config(OmegaConf.create({"trainer": {}}))
    trainer = cfg.get_trainer(mode="inference")

    # Should get inference defaults
    assert trainer.num_devices == 1


def test_strip_training_config_sections():
    """Test that CLI helper strips training-only sections."""
    from dreem.cli import _strip_training_config_sections

    cfg = OmegaConf.create(
        {
            "model": {"d_model": 128},
            "tracker": {"window_size": 8},
            "trainer": {"accelerator": "auto"},
            "dataset": {"test_dataset": {}},
            "optimizer": {"lr": 0.001},
            "scheduler": {"type": "ReduceLROnPlateau"},
            "loss": {"neg_unmatched": True},
            "early_stopping": {"patience": 5},
            "checkpointing": {"monitor": "val_loss"},
            "logging": {"name": "test"},
            "runner": {"metrics": True},
        }
    )

    result = _strip_training_config_sections(cfg)

    # Training-only sections should be removed
    for section in _TRAINING_ONLY_CONFIG_SECTIONS:
        assert section not in result, f"{section} should have been stripped"

    # Non-training sections should be preserved
    assert "model" in result
    assert "tracker" in result
    assert "trainer" in result
    assert "dataset" in result


def test_resolve_accelerator():
    """Test _resolve_accelerator handles --device/--gpu conflicts."""
    import warnings

    from dreem.cli import _resolve_accelerator

    # --device only (no --gpu)
    assert _resolve_accelerator("auto", None) == "auto"
    assert _resolve_accelerator("mps", None) == "mps"
    assert _resolve_accelerator("cpu", None) == "cpu"

    # --gpu only (deprecated, --device at default "auto")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _resolve_accelerator("auto", True) == "gpu"
        assert any(issubclass(x.category, FutureWarning) for x in w)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _resolve_accelerator("auto", False) == "cpu"
        assert any(issubclass(x.category, FutureWarning) for x in w)

    # Both --device and --gpu: --device takes precedence with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _resolve_accelerator("mps", True) == "mps"
        assert any(issubclass(x.category, UserWarning) for x in w)
