# to implement - config class that handles getters/setters
import hydra
import sys
import ast 
import torch
import lightning
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.training.losses import AssoLoss
from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.datasets.microscopy_dataset import MicroscopyDataset
from omegaconf import DictConfig, OmegaConf
from typing import Union
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
        
        def get_dataset(self, type: str = "sleap", mode: str = None) -> Union[SleapDataset, MicroscopyDataset]:
            """
            Getter for datasets
            Returns: Either a `SleapDataset` or `MicroscopyDataset` with params indicated by cfg
            Args:
                type: Either "sleap" or "microscopy". Whether to return a `SleapDataset` or `MicroscopyDataset`
                mode: [None, "train", "test", "val"]. Indicates whether to use train, val, or test params for dataset
            """
            if mode is None:
                dataset_params = self.cfg.dataset
            elif mode.lower() == "train":
                dataset_params = self.cfg.train_dataset
            elif mode.lower() == "val":
                dataset_params = self.cfg.val_dataset
            elif mode.lower() == "test":
                dataset_params = self.cfg.test_dataset
            else:
                raise ValueError("`mode` must be one of ['train', 'val','test', not '{mode}'")
            if type.lower() == "sleap":
                return SleapDataset(**dataset_params)
            elif type.lower() == "microscopy":
                return MicroscopyDataset(**dataset_params)
            else:
                raise ValueError(f"`type` must be one of ['sleap', 'microscopy'] not '{type}'!")
            
        def get_optimizer(self) -> torch.optim.Optimizer:
            """
            Getter for optimizer
            Returns: A torch Optimizer with specified params
            """
            pass

        def get_loss(self) -> AssoLoss:
            """
            Getter for loss functions
            Returns: An AssoLoss with specified params
            """
            pass

        def get_logger(self) -> lightning.pytorch.loggers.logger:
            """
            Getter for lightning logging callbacks
            Returns: A lightning Logger with specified params
            """
            pass

        def get_trainer(self) -> lightning.Trainer:
            """
            Getter for the lightning trainer:
            Returns a lightning Trainer with specified params
            """
            pass