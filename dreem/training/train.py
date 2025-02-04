"""Training script for training model.

Used for training a single model or deploying a batch train job on RUNAI CLI
"""

from dreem.io import Config
from dreem.datasets import TrackingDataset
from dreem.datasets.data_utils import view_training_batch
from multiprocessing import cpu_count
from omegaconf import DictConfig
import subprocess
import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import logging

logger = logging.getLogger("training")

class GPUMonitorCallback(pl.Callback):
    def __init__(self, log_file=None):
        super().__init__()
        if log_file is None:
            from datetime import datetime
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f'gpu_usage_{current_time}.log'
        self.log_file = log_file
        
    def log_gpu_usage(self, phase):
        # Get memory usage from PyTorch
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        max_memory_allocated = torch.cuda.max_memory_allocated()

        # Get GPU utilization and memory usage using nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            first_gpu_info = result.stdout.decode('utf-8').strip().split('\n')[0]
            gpu_utilization, memory_used, memory_total = first_gpu_info.split(', ')
        except subprocess.CalledProcessError as e:
            gpu_utilization = memory_used = memory_total = "Error retrieving data"

        # Prepare the log message
        log_message = (
            f"{phase} Phase:\n"
            f"PyTorch Memory Allocated: {memory_allocated / (1024 ** 2):.2f} MB\n"
            f"PyTorch Memory Reserved: {memory_reserved / (1024 ** 2):.2f} MB\n"
            f"PyTorch Max Memory Allocated: {max_memory_allocated / (1024 ** 2):.2f} MB\n"
            f"nvidia-smi GPU Utilization: {gpu_utilization} %\n"
            f"nvidia-smi Memory Used: {memory_used} MB\n"
            f"nvidia-smi Total Memory: {memory_total} MB\n"
        )

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(log_message)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_info = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} - Training\n"
        with open(self.log_file, 'a') as f:
            f.write(epoch_info)
        self.log_gpu_usage("Training")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_info = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} - Validation\n"
        with open(self.log_file, 'a') as f:
            f.write(epoch_info)
        self.log_gpu_usage("Validation")



@hydra.main(config_path=None, config_name=None, version_base=None)
def run(cfg: DictConfig):
    """Train model based on config.

    Handles all config parsing and initialization then calls `trainer.train()`.
    If `batch_config` is included then run will be assumed to be a batch job.

    Args:
        cfg: The config dict parsed by `hydra`
    """
    torch.set_float32_matmul_precision("medium")
    train_cfg = Config(cfg)

    # update with parameters for batch train job
    if "batch_config" in cfg.keys():
        try:
            index = int(os.environ["POD_INDEX"])
        except KeyError as e:
            index = int(
                input(f"{e}. Assuming single run!\nPlease input task index to run:")
            )

        hparams_df = pd.read_csv(cfg.batch_config)
        hparams = hparams_df.iloc[index].to_dict()

        if train_cfg.set_hparams(hparams):
            logger.debug("Updated the following hparams to the following values")
            logger.debug(hparams)
    else:
        hparams = {}
    logging.getLogger().setLevel(level=cfg.get("log_level", "INFO").upper())
    logger.info(f"Final train config: {train_cfg}")

    model = train_cfg.get_model()
    train_dataset = train_cfg.get_dataset(mode="train")
    train_dataloader = train_cfg.get_dataloader(train_dataset, mode="train")

    val_dataset = train_cfg.get_dataset(mode="val")
    val_dataloader = train_cfg.get_dataloader(val_dataset, mode="val")

    test_dataset = train_cfg.get_dataset(mode="test")
    test_dataloader = train_cfg.get_dataloader(test_dataset, mode="test")

    dataset = TrackingDataset(
        train_dl=train_dataloader, val_dl=val_dataloader, test_dl=test_dataloader
    )

    if cfg.view_batch.enable:
        instances = next(iter(train_dataset))
        view_training_batch(instances, num_frames=cfg.view_batch.num_frames)

        if cfg.view_batch.no_train:
            return

    model = train_cfg.get_gtr_runner()  # TODO see if we can use torch.compile()

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
    _ = callbacks.extend(train_cfg.get_checkpointing())
    _ = callbacks.append(pl.callbacks.LearningRateMonitor())
    _ = callbacks.append(GPUMonitorCallback())

    early_stopping = train_cfg.get_early_stopping()
    if early_stopping is not None:
        callbacks.append(early_stopping)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if torch.cuda.is_available() else cpu_count()

    trainer = train_cfg.get_trainer(
        callbacks,
        run_logger,
        accelerator=accelerator,
        devices=devices,
    )

    trainer.fit(model, dataset)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py --config-dir=./configs --config-name=base

    # override with params config:
    # python train.py --config-dir=./configs --config-name=base +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python train.py --config-dir=./configs --config-name=base +params_config=configs/params.yaml model.norm=True model.decoder_self_attn=True dataset.padding=10

    # deploy batch train job:
    # python train.py --config-dir=./configs --config-name=base +batch_config=test_batch_train.csv

    run()
