"""API for evaluating model against ground truth."""

import logging

from omegaconf import DictConfig

from dreem.io import Config
from dreem.models import GTRRunner

logger = logging.getLogger("dreem.inference")


def run(cfg: DictConfig) -> None:
    """Run evaluation based on config.

    Args:
        cfg: A DictConfig containing checkpoint path and data configuration
    """
    eval_cfg = Config(cfg)
    checkpoint = eval_cfg.get("ckpt_path", None)
    if checkpoint is None:
        raise ValueError("Checkpoint path not found in config")

    model = GTRRunner.load_from_checkpoint(checkpoint, strict=False)
    overrides_dict = model.setup_tracking(eval_cfg, mode="eval")
    logger.info(
        f"Saving tracking results and metrics to {model.test_results['save_path']}"
    )

    labels_files, vid_files = eval_cfg.get_data_paths(
        "test", eval_cfg.cfg.dataset.test_dataset
    )
    trainer = eval_cfg.get_trainer()
    for label_file, vid_file in zip(labels_files, vid_files):
        dataset = eval_cfg.get_dataset(
            label_files=[label_file],
            vid_files=[vid_file],
            mode="test",
            overrides=overrides_dict,
        )
        dataloader = eval_cfg.get_dataloader(dataset, mode="test")
        _ = trainer.test(model, dataloader)

    logger.info("Evaluation complete.")
