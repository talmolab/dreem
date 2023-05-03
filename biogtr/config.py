# to implement - config class that handles getters/setters
import hydra
import sys
import ast
import torch
import pytorch_lightning as pl
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.training.losses import AssoLoss
from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.datasets.microscopy_dataset import MicroscopyDataset
from omegaconf import DictConfig, OmegaConf
from typing import Union, Iterable

"""
Class for handling config parsing
"""


class Config:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the class with config from hydra/omega conf
        First uses `base_param` file then overwrites with specific `params_config`
        Args:
            cfg: The `DictConfig` containing all the hyperparameters needed for training/evaluation
        """
        base_cfg = OmegaConf.load(cfg.base_config)

        if "params_config" in cfg:
            # merge configs
            params_config = OmegaConf.load(cfg.params_config)
            self.cfg = OmegaConf.merge(base_cfg, params_config)
        else:
            # just use base config
            self.cfg = base_cfg

    def set_hparams(self, hparams: dict):
        """
        Setter function for overwriting specific hparams.
        Useful for changing 1 or 2 hyperparameters such as dataset
        Args:
            hparams: A dict containing the hyperparameter to be overwritten and the value to be changed to
        """
        for hparam, val in hparams.items():
            OmegaConf.update(self.cfg, hparam, val)

    def get_model(self) -> GlobalTrackingTransformer:
        """
        Getter for gtr model
        Returns: A global tracking transformer with parameters indicated by cfg
        """
        model_params = self.cfg.model
        return GlobalTrackingTransformer(**model_params)

    def get_dataset(
        self, type: str, mode: str
    ) -> Union[SleapDataset, MicroscopyDataset]:
        """
        Getter for datasets
        Returns: Either a `SleapDataset` or `MicroscopyDataset` with params indicated by cfg
        Args:
            type: Either "sleap" or "microscopy". Whether to return a `SleapDataset` or `MicroscopyDataset`
            mode: [None, "train", "test", "val"]. Indicates whether to use train, val, or test params for dataset
        """
        if mode.lower() == "train":
            dataset_params = self.cfg.dataset.train_dataset
        elif mode.lower() == "val":
            dataset_params = self.cfg.dataset.val_dataset
        elif mode.lower() == "test":
            dataset_params = self.cfg.dataset.test_dataset
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )
        if type.lower() == "sleap":
            return SleapDataset(**dataset_params)
        elif type.lower() == "microscopy":
            return MicroscopyDataset(**dataset_params)
        else:
            raise ValueError(
                f"`type` must be one of ['sleap', 'microscopy'] not '{type}'!"
            )

    def get_dataloader(
        self, dataset: Union[SleapDataset, MicroscopyDataset], mode: str
    ) -> torch.utils.data.DataLoader:
        if mode.lower() == "train":
            dataloader_params = self.cfg.dataset.train_dataloader
        elif mode.lower() == "val":
            dataloader_params = self.cfg.dataset.val_dataloader
        elif mode.lower() == "test":
            dataloader_params = self.cfg.dataset.test_dataloader
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )
        return torch.utils.data.DataLoader(
            dataset,
            pin_memory=True if dataloader_params.num_workers > 0 else False,
            generator=torch.Generator(device="cuda")
            if torch.cuda.is_available()
            else None,
            **dataloader_params,
        )

    def get_optimizer(self, params: Iterable) -> torch.optim.Optimizer:
        """
        Getter for optimizer
        Returns: A torch Optimizer with specified params
        Args:
            params: iterable of model parameters to optimize or dicts defining parameter groups
        """
        optimizer_params = self.cfg.optimizer
        return torch.optim.Adam(params=params, **optimizer_params)

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Getter for lr scheduler
        Returns a torch learning rate scheduler with specified params
        Args:
            optimizer: The optimizer to wrap the scheduler around
        """
        lr_scheduler_params = self.cfg.scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **lr_scheduler_params
        )

    def get_loss(self) -> AssoLoss:
        """
        Getter for loss functions
        Returns: An AssoLoss with specified params
        """
        loss_params = self.cfg.loss
        return AssoLoss(**loss_params)

    def get_logger(self) -> pl.loggers.WandbLogger:
        """
        Getter for lightning logging callback
        Returns: A lightning Logger with specified params
        """
        logger_params = self.cfg.logging
        return pl.loggers.WandbLogger(**logger_params)

    def get_early_stopping(self) -> pl.callbacks.EarlyStopping:
        """
        Getter for lightning early stopping callbac
        Returns: a lightning early stopping callback with specified params
        """
        early_stopping_params = self.cfg.early_stopping
        return pl.callbacks.EarlyStopping(**early_stopping_params)

    def get_checkpointing(self, dirpath: str) -> pl.callbacks.ModelCheckpoint:
        """
        getter for lightning checkpointing callback
        Returns: a lightning checkpointing callback with specified params
        Args:
            dirpath: the path to the directory where checkpoints will be stored
        """
        checkpoint_params = self.cfg.checkpointing
        return pl.callbacks.ModelCheckpoint(dirpath=dirpath, **checkpoint_params)

    def get_trainer(self, callbacks: list[pl.callbacks.Callback]) -> pl.Trainer:
        """
        Getter for the lightning trainer:
        Returns a lightning Trainer with specified params
        Args:
            callbacks: a list of lightning callbacks preconfigured to be used for training
        """
        trainer_params = self.cfg.trainer_params
        return pl.Trainer(callbacks=callbacks, **trainer_params)
