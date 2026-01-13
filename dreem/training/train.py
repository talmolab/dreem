"""API for training models."""

import logging

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from dreem.datasets import TrackingDataset
from dreem.io import Config

logger = logging.getLogger("dreem.training")


def run(cfg: DictConfig) -> None:
    """Train model based on config.

    Args:
        cfg: A DictConfig containing training configuration
    """
    torch.set_float32_matmul_precision("medium")
    train_cfg = Config(cfg)

    logger.info(f"Training config: {train_cfg}")

    train_dataset = train_cfg.get_dataset(mode="train")
    train_dataloader = train_cfg.get_dataloader(train_dataset, mode="train")

    val_dataset = train_cfg.get_dataset(mode="val")
    val_dataloader = train_cfg.get_dataloader(val_dataset, mode="val")

    dataset = TrackingDataset(train_dl=train_dataloader, val_dl=val_dataloader)

    model = train_cfg.get_gtr_runner()

    run_logger = train_cfg.get_logger()
    if run_logger is not None and isinstance(run_logger, pl.loggers.wandb.WandbLogger):
        data_paths = train_cfg.data_paths
        flattened_paths = [
            [item] for sublist in data_paths.values() for item in sublist
        ]
        run_logger.log_text(
            "training_files", columns=["data_paths"], data=flattened_paths
        )

    callbacks = []
    callbacks.extend(train_cfg.get_checkpointing())
    callbacks.append(pl.callbacks.LearningRateMonitor())

    early_stopping = train_cfg.get_early_stopping()
    if early_stopping is not None:
        callbacks.append(early_stopping)

    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    trainer = train_cfg.get_trainer(
        callbacks,
        run_logger,
        devices=devices,
    )

    logger.info("Starting training...")
    trainer.fit(model, dataset)
    logger.info("Training complete.")
