"""Tests for `config.py`"""

import torch
from omegaconf import OmegaConf, open_dict

from dreem.io import Config
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
