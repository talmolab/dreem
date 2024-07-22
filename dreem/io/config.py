# to implement - config class that handles getters/setters
"""Data structures for handling config parsing."""

from omegaconf import DictConfig, OmegaConf, open_dict
from pprint import pprint
from typing import Union, Iterable
from pathlib import Path
import glob
import pytorch_lightning as pl
import torch


class Config:
    """Class handling loading components based on config params."""

    def __init__(self, cfg: DictConfig, params_cfg: DictConfig = None):
        """Initialize the class with config from hydra/omega conf.

        First uses `base_param` file then overwrites with specific `params_config`.

        Args:
            cfg: The `DictConfig` containing all the hyperparameters needed for
                training/evaluation.
            params_cfg: The `DictConfig` containing subset of hyperparameters to override.
                training/evaluation
        """
        base_cfg = cfg
        print(f"Base Config: {cfg}")

        if "params_config" in cfg:
            params_cfg = OmegaConf.load(cfg.params_config)

        if params_cfg:
            pprint(f"Overwriting base config with {params_cfg}")
            with open_dict(base_cfg):
                self.cfg = OmegaConf.merge(base_cfg, params_cfg)  # merge configs
        else:
            self.cfg = cfg

    def __repr__(self):
        """Object representation of config class."""
        return f"Config({self.cfg})"

    def __str__(self):
        """Return a string representation of config class."""
        return f"Config({self.cfg})"

    @classmethod
    def from_yaml(cls, base_cfg_path: str, params_cfg_path: str = None) -> None:
        """Load config directly from yaml.

        Args:
            base_cfg_path: path to base config file.
            params_cfg_path: path to override params.
        """
        base_cfg = OmegaConf.load(base_cfg_path)
        params_cfg = OmegaConf.load(params_cfg_path) if params_cfg else None
        return cls(base_cfg, params_cfg)

    def set_hparams(self, hparams: dict) -> bool:
        """Setter function for overwriting specific hparams.

        Useful for changing 1 or 2 hyperparameters such as dataset.

        Args:
            hparams: A dict containing the hyperparameter to be overwritten and
                the value to be changed

        Returns:
            `True` if config is successfully updated, `False` otherwise
        """
        if hparams == {} or hparams is None:
            print("Nothing to update!")
            return False
        for hparam, val in hparams.items():
            try:
                OmegaConf.update(self.cfg, hparam, val)
            except Exception as e:
                print(f"Failed to update {hparam} to {val} due to {e}")
                return False
        return True

    def get_model(self) -> "GlobalTrackingTransformer":
        """Getter for gtr model.

        Returns:
            A global tracking transformer with parameters indicated by cfg
        """
        from dreem.models import GlobalTrackingTransformer

        model_params = OmegaConf.to_container(self.cfg.model)
        ckpt_path = model_params.pop("ckpt_path", None)

        if ckpt_path is not None and len(ckpt_path) > 0:
            return GTRRunner.load_from_checkpoint(ckpt_path).model

        return GlobalTrackingTransformer(**model_params)

    def get_tracker_cfg(self) -> dict:
        """Getter for tracker config params.

        Returns:
            A dict containing the init params for `Tracker`.
        """
        tracker_params = self.cfg.tracker
        tracker_cfg = {}
        for key, val in tracker_params.items():
            tracker_cfg[key] = val
        return tracker_cfg

    def get_gtr_runner(self) -> "GTRRunner":
        """Get lightning module for training, validation, and inference."""
        from dreem.models import GTRRunner

        tracker_params = self.cfg.tracker
        optimizer_params = self.cfg.optimizer
        scheduler_params = self.cfg.scheduler
        loss_params = self.cfg.loss
        gtr_runner_params = self.cfg.runner
        model_params = OmegaConf.to_container(self.cfg.model)

        ckpt_path = model_params.pop("ckpt_path", None)

        if ckpt_path is not None and ckpt_path != "":
            model = GTRRunner.load_from_checkpoint(
                ckpt_path,
                tracker_cfg=tracker_params,
                train_metrics=self.cfg.runner.metrics.train,
                val_metrics=self.cfg.runner.metrics.val,
                test_metrics=self.cfg.runner.metrics.test,
            )

        else:
            model = GTRRunner(
                model_params,
                tracker_params,
                loss_params,
                optimizer_params,
                scheduler_params,
                **gtr_runner_params,
            )

        return model

    def get_data_paths(self, data_cfg: dict) -> tuple[list[str], list[str]]:
        """Get file paths from directory.

        Args:
            data_cfg: Config for the dataset containing "dir" key.

        Returns:
            lists of labels file paths and video file paths respectively
        """
        dir_cfg = data_cfg.pop("dir", None)

        if dir_cfg:
            labels_suff = dir_cfg.labels_suffix
            vid_suff = dir_cfg.vid_suffix
            labels_path = f"{dir_cfg.path}/*{labels_suff}"
            vid_path = f"{dir_cfg.path}/*{vid_suff}"
            print(f"Searching for labels matching {labels_path}")
            label_files = glob.glob(labels_path)
            print(f"Searching for videos matching {vid_path}")
            vid_files = glob.glob(vid_path)
            print(f"Found {len(label_files)} labels and {len(vid_files)} videos")
            return label_files, vid_files

        return None, None

    def get_dataset(
        self, mode: str
    ) -> Union["SleapDataset", "MicroscopyDataset", "CellTrackingDataset"]:
        """Getter for datasets.

        Args:
            mode: [None, "train", "test", "val"]. Indicates whether to use
                train, val, or test params for dataset

        Returns:
            Either a `SleapDataset` or `MicroscopyDataset` with params indicated by cfg
        """
        from dreem.datasets import MicroscopyDataset, SleapDataset, CellTrackingDataset

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
        dataset_params = OmegaConf.to_container(dataset_params)
        label_files, vid_files = self.get_data_paths(dataset_params)
        # todo: handle this better
        if "slp_files" in dataset_params:
            if label_files is not None:
                dataset_params.slp_files = label_files
            if vid_files is not None:
                dataset_params.video_files = vid_files
            return SleapDataset(**dataset_params)

        elif "tracks" in dataset_params or "source" in dataset_params:
            if label_files is not None:
                dataset_params.tracks = label_files
            if vid_files is not None:
                dataset_params.video_files = vid_files
            return MicroscopyDataset(**dataset_params)

        elif "raw_images" in dataset_params:
            if label_files is not None:
                dataset_params.gt_images = label_files
            if vid_files is not None:
                dataset_params.raw_images = vid_files
            return CellTrackingDataset(**dataset_params)

        # todo: handle this better
        if "slp_files" in dataset_params:
            return SleapDataset(**dataset_params)

        elif "tracks" in dataset_params or "source" in dataset_params:
            return MicroscopyDataset(**dataset_params)

        elif "raw_images" in dataset_params:
            return CellTrackingDataset(**dataset_params)

        else:
            raise ValueError(
                "Could not resolve dataset type from Config! Please include \
                either `slp_files` or `tracks`/`source`"
            )

    def get_dataloader(
        self,
        dataset: Union["SleapDataset", "MicroscopyDataset", "CellTrackingDataset"],
        mode: str,
    ) -> torch.utils.data.DataLoader:
        """Getter for dataloader.

        Args:
            dataset: the Sleap or Microscopy Dataset used to initialize the dataloader
            mode: either ["train", "val", or "test"] indicates which dataset
                config to use

        Returns:
            A torch dataloader for `dataset` with parameters configured as specified
        """
        if mode.lower() == "train":
            dataloader_params = self.cfg.dataloader.train_dataloader
        elif mode.lower() == "val":
            dataloader_params = self.cfg.dataloader.val_dataloader
        elif mode.lower() == "test":
            dataloader_params = self.cfg.dataloader.test_dataloader
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )
        if dataloader_params.num_workers > 0:
            # prevent too many open files error
            pin_memory = True
            torch.multiprocessing.set_sharing_strategy("file_system")
        else:
            pin_memory = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            pin_memory=pin_memory,
            collate_fn=dataset.no_batching_fn,
            **dataloader_params,
        )

    def get_optimizer(self, params: Iterable) -> torch.optim.Optimizer:
        """Getter for optimizer.

        Args:
            params: iterable of model parameters to optimize or dicts defining
                parameter groups

        Returns:
            A torch Optimizer with specified params
        """
        from dreem.models.model_utils import init_optimizer

        optimizer_params = self.cfg.optimizer

        return init_optimizer(params, optimizer_params)

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Getter for lr scheduler.

        Args:
            optimizer: The optimizer to wrap the scheduler around

        Returns:
            A torch learning rate scheduler with specified params
        """
        from dreem.models.model_utils import init_scheduler

        lr_scheduler_params = self.cfg.scheduler

        return init_scheduler(optimizer, lr_scheduler_params)

    def get_loss(self) -> "dreem.training.losses.AssoLoss":
        """Getter for loss functions.

        Returns:
            An AssoLoss with specified params
        """
        from dreem.training.losses import AssoLoss

        loss_params = self.cfg.loss

        return AssoLoss(**loss_params)

    def get_logger(self) -> pl.loggers.Logger:
        """Getter for logging callback.

        Returns:
            A Logger with specified params
        """
        from dreem.models.model_utils import init_logger

        logger_params = OmegaConf.to_container(self.cfg.logging, resolve=True)

        return init_logger(
            logger_params, OmegaConf.to_container(self.cfg, resolve=True)
        )

    def get_early_stopping(self) -> pl.callbacks.EarlyStopping:
        """Getter for lightning early stopping callback.

        Returns:
            A lightning early stopping callback with specified params
        """
        early_stopping_params = self.cfg.early_stopping
        return pl.callbacks.EarlyStopping(**early_stopping_params)

    def get_checkpointing(self) -> pl.callbacks.ModelCheckpoint:
        """Getter for lightning checkpointing callback.

        Returns:
            A lightning checkpointing callback with specified params
        """
        # convert to dict to enable extracting/removing params
        checkpoint_params = OmegaConf.to_container(self.cfg.checkpointing, resolve=True)
        logging_params = self.cfg.logging
        dirpath = checkpoint_params.pop("dirpath", None)
        if dirpath is None:
            if "group" in logging_params:
                dirpath = f"./models/{logging_params.group}/{logging_params.name}"
            else:
                dirpath = f"./models/{logging_params.name}"

        dirpath = Path(dirpath).resolve()
        if not Path(dirpath).exists():
            try:
                Path(dirpath).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(
                    f"Cannot create a new folder. Check the permissions to the given Checkpoint directory. \n {e}"
                )

        checkpointers = []
        monitor = checkpoint_params.pop("monitor")
        for metric in monitor:
            checkpointer = pl.callbacks.ModelCheckpoint(
                monitor=metric,
                dirpath=dirpath,
                filename=f"{{epoch}}-{{{metric}}}",
                **checkpoint_params,
            )
            checkpointer.CHECKPOINT_NAME_LAST = f"{{epoch}}-best-{{{metric}}}"
            checkpointers.append(checkpointer)
        return checkpointers

    def get_trainer(
        self,
        callbacks: list[pl.callbacks.Callback] = None,
        logger: pl.loggers.WandbLogger = None,
        devices: int = 1,
        accelerator: str = "auto",
    ) -> pl.Trainer:
        """Getter for the lightning trainer.

        Args:
            callbacks: a list of lightning callbacks preconfigured to be used
                for training
            logger: the Wandb logger used for logging during training
            devices: The number of gpus to be used. 0 means cpu
            accelerator: either "gpu" or "cpu" specifies which device to use

        Returns:
            A lightning Trainer with specified params
        """
        if "trainer" in self.cfg:
            trainer_params = self.cfg.trainer

        else:
            trainer_params = {}

        profiler = trainer_params.pop("profiler", None)
        if "profiler":
            profiler = pl.profilers.AdvancedProfiler(filename="profile.txt")
        else:
            profiler = None

        if "accelerator" not in trainer_params:
            trainer_params["accelerator"] = accelerator
        if "devices" not in trainer_params:
            trainer_params["devices"] = devices

        return pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            profiler=profiler,
            **trainer_params,
        )
