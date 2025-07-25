"""Script to evaluate model."""

import logging
import os

import hydra
import pandas as pd
import sleap_io as sio
from omegaconf import DictConfig

from dreem.inference import BatchTracker, Tracker
from dreem.io import Config
from dreem.models import GTRRunner

logger = logging.getLogger("dreem.inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)


@hydra.main(config_path=None, config_name=None, version_base=None)
def run(cfg: DictConfig) -> dict[int, sio.Labels]:
    """Run inference based on config file.

    Args:
        cfg: A dictconfig loaded from hydra containing checkpoint path and data
    """
    eval_cfg = Config(cfg)

    if "checkpoints" in cfg.keys():
        try:
            index = int(os.environ["POD_INDEX"])
        # For testing without deploying a job on runai
        except KeyError:
            index = input("Pod Index Not found! Please choose a pod index: ")

        logger.info(f"Pod Index: {index}")

        checkpoints = pd.read_csv(cfg.checkpoints)
        checkpoint = checkpoints.iloc[index]
    else:
        checkpoint = eval_cfg.get("ckpt_path", None)
        if checkpoint is None:
            raise ValueError("Checkpoint path not found in config")

    logging.getLogger().setLevel(level=cfg.get("log_level", "INFO").upper())

    model = GTRRunner.load_from_checkpoint(checkpoint, strict=False)
    model.tracker_cfg = eval_cfg.cfg.tracker
    if model.tracker_cfg.get("tracker_type", "standard") == "batch":
        model.tracker = BatchTracker(**model.tracker_cfg)
    else:
        model.tracker = Tracker(**model.tracker_cfg)
    logger.info("Using the following tracker:")
    logger.info(model.tracker)
    model.metrics["test"] = eval_cfg.get("metrics", {}).get("test", "all")
    model.persistent_tracking["test"] = True
    logger.info("Computing the following metrics:")
    logger.info(model.metrics["test"])
    model.test_results["save_path"] = eval_cfg.get("outdir", ".")
    os.makedirs(model.test_results["save_path"], exist_ok=True)
    logger.info(
        f"Saving tracking results and metrics to {model.test_results['save_path']}"
    )

    labels_files, vid_files = eval_cfg.get_data_paths(
        "test", eval_cfg.cfg.dataset.test_dataset
    )
    trainer = eval_cfg.get_trainer()
    for label_file, vid_file in zip(labels_files, vid_files):
        dataset = eval_cfg.get_dataset(
            label_files=[label_file], vid_files=[vid_file], mode="test"
        )
        dataloader = eval_cfg.get_dataloader(dataset, mode="test")
        metrics = trainer.test(model, dataloader)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python eval.py --config-dir=./configs --config-name=inference

    # override with params config:
    # python eval.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python eval.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml dataset.train_dataset.padding=10
    run()
