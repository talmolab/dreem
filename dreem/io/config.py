# to implement - config class that handles getters/setters
"""Data structures for handling config parsing."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

if TYPE_CHECKING:
    from dreem.datasets import CellTrackingDataset, MicroscopyDataset, SleapDataset
    from dreem.models import GlobalTrackingTransformer, GTRRunner
    from dreem.training.losses import AssoLoss

logger = logging.getLogger("dreem.io")


class Config:
    """Class handling loading components based on config params."""

    def __init__(self, cfg: DictConfig, params_cfg: DictConfig | None = None):
        """Initialize the class with config from hydra/omega conf.

        First uses `base_param` file then overwrites with specific `params_config`.

        Args:
            cfg: The `DictConfig` containing all the hyperparameters needed for
                training/evaluation.
            params_cfg: The `DictConfig` containing subset of hyperparameters to override.
                training/evaluation
        """
        base_cfg = cfg
        logger.info(f"Base Config: {cfg}")

        if "params_config" in cfg:
            params_cfg = OmegaConf.load(cfg.params_config)

        if params_cfg:
            logger.info(f"Overwriting base config with {params_cfg}")
            with open_dict(base_cfg):
                self.cfg = OmegaConf.merge(base_cfg, params_cfg)  # merge configs
        else:
            self.cfg = cfg

        OmegaConf.set_struct(self.cfg, False)

        self._vid_files = {}

    def __repr__(self):
        """Object representation of config class."""
        return f"Config({self.cfg})"

    def __str__(self):
        """Return a string representation of config class."""
        return f"Config({self.cfg})"

    @classmethod
    def from_yaml(cls, base_cfg_path: str, params_cfg_path: str | None = None) -> None:
        """Load config directly from yaml.

        Args:
            base_cfg_path: path to base config file.
            params_cfg_path: path to override params.
        """
        base_cfg = OmegaConf.load(base_cfg_path)
        params_cfg = OmegaConf.load(params_cfg_path) if params_cfg_path else None
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
            logger.warning("Nothing to update!")
            return False
        for hparam, val in hparams.items():
            try:
                OmegaConf.update(self.cfg, hparam, val)
            except Exception as e:
                logger.exception(f"Failed to update {hparam} to {val} due to {e}")
                return False
        return True

    def get(self, key: str, default=None, cfg: dict = None):
        """Get config item.

        Args:
            key: key of item to return
            default: default value to return if key is missing.
            cfg: the config dict from which to retrieve an item
        """
        if cfg is None:
            cfg = self.cfg

        param = cfg.get(key, default)

        if isinstance(param, DictConfig):
            param = OmegaConf.to_container(param, resolve=True)

        return param

    def get_model(self) -> GlobalTrackingTransformer:
        """Getter for gtr model.

        Returns:
            A global tracking transformer with parameters indicated by cfg
        """
        from dreem.models import GlobalTrackingTransformer, GTRRunner

        model_params = self.get("model", {})

        ckpt_path = model_params.pop("ckpt_path", None)

        if ckpt_path is not None and len(ckpt_path) > 0:
            return GTRRunner.load_from_checkpoint(ckpt_path).model

        return GlobalTrackingTransformer(**model_params)

    def get_tracker_cfg(self) -> dict:
        """Getter for tracker config params.

        Returns:
            A dict containing the init params for `Tracker`.
        """
        return self.get("tracker", {})

    def get_gtr_runner(self, ckpt_path: str | None = None) -> GTRRunner:
        """Get lightning module for training, validation, and inference.

        Args:
            ckpt_path: path to checkpoint for override

        Returns:
            a gtr runner model
        """
        from dreem.models import GTRRunner

        keys = ["tracker", "optimizer", "scheduler", "loss", "runner", "model"]
        args = [key + "_cfg" if key != "runner" else key for key in keys]

        params = {}
        for key, arg in zip(keys, args):
            sub_params = self.get(key, {})

            if len(sub_params) == 0:
                logger.warning(
                    f"`{key}` not found in config or is empty. Using defaults for {arg}!"
                )

            if key == "runner":
                runner_params = sub_params
                for k, v in runner_params.items():
                    params[k] = v
            else:
                params[arg] = sub_params

        ckpt_path = params["model_cfg"].pop("ckpt_path", None)

        if ckpt_path is not None and ckpt_path != "":
            model = GTRRunner.load_from_checkpoint(
                ckpt_path, tracker_cfg=params["tracker_cfg"], **runner_params
            )

        else:
            model = GTRRunner(**params)

        return model

    def get_ctc_paths(
        self, list_dir_path: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Get file paths from directory. Only for CTC datasets.

        Args:
            list_dir_path: list of directories to search for labels and videos

        Returns:
            lists of labels file paths and video file paths
        """
        gt_list = []
        raw_img_list = []
        ctc_track_meta = []
        # user can specify a list of directories, each of which can contain several subdirectories that come in pairs of (dset_name, dset_name_GT/TRA)
        for dir_path in list_dir_path:
            for subdir in os.listdir(dir_path):
                if subdir.endswith("_GT"):
                    gt_path = os.path.join(dir_path, subdir, "TRA")
                    raw_img_path = os.path.join(dir_path, subdir.replace("_GT", ""))
                    # get filepaths for all tif files in gt_path
                    gt_list.append(glob.glob(os.path.join(gt_path, "*.tif")))
                    # get filepaths for all tif files in raw_img_path
                    raw_img_list.append(glob.glob(os.path.join(raw_img_path, "*.tif")))
                    man_track_file = glob.glob(os.path.join(gt_path, "man_track.txt"))
                    if len(man_track_file) > 0:
                        ctc_track_meta.append(man_track_file[0])
                    else:
                        logger.debug(
                            f"No man_track.txt file found in {gt_path}. Continuing..."
                        )
                else:
                    continue

        return gt_list, raw_img_list, ctc_track_meta

    def get_data_paths(self, mode: str, data_cfg: dict) -> tuple[list[str], list[str]]:
        """Get file paths from directory. Only for SLEAP datasets.

        Args:
            mode: [None, "train", "test", "val"]. Indicates whether to use
                train, val, or test params for dataset
            data_cfg: Config for the dataset containing "dir" key.

        Returns:
            lists of labels file paths and video file paths respectively
        """
        # hack to get around the fact that for test mode, get_data_paths is called before get_dataset.
        # also, for train/val mode, data_cfg has had the dir key popped through self.get() called in get_dataset()
        if mode == "test":
            list_dir_path = data_cfg.get("dir", {}).get("path", None)
            if list_dir_path is None:
                raise ValueError(
                    "`dir` is missing from dataset config. Please provide a path to the directory containing the labels and videos."
                )
            self.labels_suffix = data_cfg.get("dir", {}).get("labels_suffix")
            self.vid_suffix = data_cfg.get("dir", {}).get("vid_suffix")
        else:
            list_dir_path = self.data_dirs
        if not isinstance(list_dir_path, list):
            list_dir_path = [list_dir_path]

        if self.labels_suffix == ".slp":
            label_files = []
            vid_files = []
            for dir_path in list_dir_path:
                logger.debug(f"Searching `{dir_path}` directory")
                labels_path = f"{dir_path}/*{self.labels_suffix}"
                vid_path = f"{dir_path}/*{self.vid_suffix}"
                logger.debug(f"Searching for labels matching {labels_path}")
                label_files.extend(glob.glob(labels_path))
                logger.debug(f"Searching for videos matching {vid_path}")
                vid_files.extend(glob.glob(vid_path))

        elif self.labels_suffix == ".tif":
            label_files, vid_files, ctc_track_meta = self.get_ctc_paths(list_dir_path)

        logger.debug(f"Found {len(label_files)} labels and {len(vid_files)} videos")

        # backdoor to set label files directly in the configs (i.e. bypass dir.path)
        if data_cfg.get("slp_files", None):
            logger.debug("Overriding label files with user provided list")
            slp_files = data_cfg.get("slp_files")
            if len(slp_files) > 0:
                label_files = slp_files
        if data_cfg.get("video_files", None):
            individual_video_files = data_cfg.get("video_files")
            if len(individual_video_files) > 0:
                vid_files = individual_video_files
        return label_files, vid_files

    def get_dataset(
        self,
        mode: str,
        label_files: list[str] | None = None,
        vid_files: list[str | list[str]] = None,
        overrides: dict | None = None,
    ) -> SleapDataset | CellTrackingDataset:
        """Getter for datasets.

        Args:
            mode: [None, "train", "test", "val"]. Indicates whether to use
                train, val, or test params for dataset
            label_files: path to label_files for override
            vid_files: path to vid_files for override
            overrides: overrides to apply to the dataset config
        Returns:
            Either a `SleapDataset` or `CellTrackingDataset` with params indicated by cfg
        """
        from dreem.datasets import CellTrackingDataset, SleapDataset

        dataset_params = self.get("dataset")
        if dataset_params is None:
            raise KeyError("`dataset` key is missing from cfg!")

        if mode.lower() == "train":
            dataset_params = self.get("train_dataset", {}, dataset_params)
        elif mode.lower() == "val":
            dataset_params = self.get("val_dataset", {}, dataset_params)
        elif mode.lower() == "test":
            dataset_params = self.get("test_dataset", {}, dataset_params)
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )

        # input validation
        self.data_dirs = dataset_params.get("dir", {}).get("path", None)
        self.labels_suffix = dataset_params.get("dir", {}).get("labels_suffix")
        self.vid_suffix = dataset_params.get("dir", {}).get("vid_suffix")
        if self.data_dirs is None:
            raise ValueError(
                "`dir` is missing from dataset config. Please provide a path to the directory containing the labels and videos."
            )
        if self.labels_suffix is None or self.vid_suffix is None:
            raise KeyError(
                f"Must provide a labels suffix and vid suffix to search for but found {self.labels_suffix} and {self.vid_suffix}"
            )

        # infer dataset type from the user provided suffix
        if self.labels_suffix == ".slp":
            # during training, multiple files can be used at once, so label_files is not passed in
            # during inference, a single label_files string can be passed in as get_data_paths is
            # called before get_dataset, hence the check
            if label_files is None or vid_files is None:
                label_files, vid_files = self.get_data_paths(mode, dataset_params)
            dataset_params["slp_files"] = label_files
            dataset_params["video_files"] = vid_files
            dataset_params["data_dirs"] = self.data_dirs
            self.data_paths = (mode, vid_files)
            if overrides:
                dataset_params.update(overrides)
            return SleapDataset(**dataset_params)

        elif self.labels_suffix == ".tif":
            # for CTC datasets, pass in a list of gt and raw image directories, eaech of which contain tifs
            ctc_track_meta = None
            list_dir_path = self.data_dirs  # don't modify self.data_dirs
            if not isinstance(list_dir_path, list):
                list_dir_path = [list_dir_path]
            if label_files is None or vid_files is None:
                label_files, vid_files, ctc_track_meta = self.get_ctc_paths(
                    list_dir_path
                )
            dataset_params["data_dirs"] = self.data_dirs
            # extract filepaths of all raw images and gt images (i.e. labelled masks)
            dataset_params["gt_list"] = label_files
            dataset_params["raw_img_list"] = vid_files
            dataset_params["ctc_track_meta"] = ctc_track_meta
            if overrides:
                dataset_params.update(overrides)
            return CellTrackingDataset(**dataset_params)

        else:
            raise ValueError(
                "Could not resolve dataset type from Config! Only .slp (SLEAP) and .tif (Cell Tracking Challenge) data formats are supported."
            )

    @property
    def data_paths(self):
        """Get data paths."""
        return self._vid_files

    @data_paths.setter
    def data_paths(self, paths: tuple[str, list[str]]):
        """Set data paths.

        Args:
            paths: A tuple containing (mode, vid_files)
        """
        mode, vid_files = paths
        self._vid_files[mode] = vid_files

    def get_dataloader(
        self,
        dataset: SleapDataset | MicroscopyDataset | CellTrackingDataset,
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
        dataloader_params = self.get("dataloader", {})
        if mode.lower() == "train":
            dataloader_params = self.get("train_dataloader", {}, dataloader_params)
        elif mode.lower() == "val":
            dataloader_params = self.get("val_dataloader", {}, dataloader_params)
        elif mode.lower() == "test":
            dataloader_params = self.get("test_dataloader", {}, dataloader_params)
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )
        if dataloader_params.get("num_workers", 0) > 0:
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

        optimizer_params = self.get("optimizer")

        return init_optimizer(params, optimizer_params)

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Getter for lr scheduler.

        Args:
            optimizer: The optimizer to wrap the scheduler around

        Returns:
            A torch learning rate scheduler with specified params
        """
        from dreem.models.model_utils import init_scheduler

        lr_scheduler_params = self.get("scheduler")

        if lr_scheduler_params is None:
            logger.warning(
                "`scheduler` key not found in cfg or is empty. No scheduler will be returned!"
            )
            return None
        return init_scheduler(optimizer, lr_scheduler_params)

    def get_loss(self) -> AssoLoss:
        """Getter for loss functions.

        Returns:
            An AssoLoss with specified params
        """
        from dreem.training.losses import AssoLoss

        loss_params = self.get("loss", {})

        if len(loss_params) == 0:
            logger.warning(
                "`loss` key not found in cfg. Using default params for `AssoLoss`"
            )

        return AssoLoss(**loss_params)

    def get_logger(self) -> pl.loggers.Logger:
        """Getter for logging callback.

        Returns:
            A Logger with specified params
        """
        from dreem.models.model_utils import init_logger

        logger_params = self.get("logging", {})
        if len(logger_params) == 0:
            logger.warning(
                "`logging` key not found in cfg. No logger will be configured!"
            )

        return init_logger(
            logger_params, OmegaConf.to_container(self.cfg, resolve=True)
        )

    def get_early_stopping(self) -> pl.callbacks.EarlyStopping:
        """Getter for lightning early stopping callback.

        Returns:
            A lightning early stopping callback with specified params
        """
        early_stopping_params = self.get("early_stopping", None)

        if early_stopping_params is None:
            logger.warning(
                "`early_stopping` was not found in cfg or was `null`. Early stopping will not be used!"
            )
            return None
        elif len(early_stopping_params) == 0:
            logger.warning("`early_stopping` cfg is empty! Using defaults")
        return pl.callbacks.EarlyStopping(**early_stopping_params)

    def get_checkpointing(self) -> pl.callbacks.ModelCheckpoint:
        """Getter for lightning checkpointing callback.

        Returns:
            A lightning checkpointing callback with specified params
        """
        # convert to dict to enable extracting/removing params
        checkpoint_params = self.get("checkpointing", {})
        logging_params = self.get("logging", {})

        dirpath = checkpoint_params.pop("dirpath", None)

        if dirpath is None:
            dirpath = f"./models/{self.get('group', '', logging_params)}/{self.get('name', '', logging_params)}"

        dirpath = Path(dirpath).resolve()
        if not Path(dirpath).exists():
            try:
                Path(dirpath).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.exception(
                    f"Cannot create a new folder!. Check the permissions to {dirpath}. \n {e}"
                )

        _ = checkpoint_params.pop("dirpath", None)
        monitor = checkpoint_params.pop("monitor", ["val_loss"])
        checkpointers = []

        logger.info(
            f"Saving checkpoints to `{dirpath}` based on the following metrics: {monitor}"
        )
        if len(checkpoint_params) == 0:
            logger.warning(
                """`checkpointing` key was not found in cfg or was empty!
                Configuring checkpointing to use default params!"""
            )

        for metric in monitor:
            checkpointer = pl.callbacks.ModelCheckpoint(
                monitor=metric,
                dirpath=dirpath,
                filename=f"{{epoch}}-{{{metric}}}",
                **checkpoint_params,
            )
            checkpointer.CHECKPOINT_NAME_LAST = f"{{epoch}}-final-{{{metric}}}"
            checkpointers.append(checkpointer)
        return checkpointers

    def get_trainer(
        self,
        callbacks: list[pl.callbacks.Callback] | None = None,
        logger: pl.loggers.WandbLogger | None = None,
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
        trainer_params = self.get("trainer", {})
        profiler = trainer_params.pop("profiler", None)
        if len(trainer_params) == 0:
            print(
                "`trainer` key was not found in cfg or was empty. Using defaults for `pl.Trainer`!"
            )

        if "accelerator" not in trainer_params:
            trainer_params["accelerator"] = accelerator
        if "devices" not in trainer_params:
            trainer_params["devices"] = devices

        map_profiler = {
            "advanced": pl.profilers.AdvancedProfiler,
            "simple": pl.profilers.SimpleProfiler,
            "pytorch": pl.profilers.PyTorchProfiler,
            "passthrough": pl.profilers.PassThroughProfiler,
            "xla": pl.profilers.XLAProfiler,
        }

        if profiler:
            if profiler in map_profiler:
                profiler = map_profiler[profiler](filename="profile")
            else:
                raise ValueError(
                    f"Profiler {profiler} not supported! Please use one of {list(map_profiler.keys())}"
                )

        return pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            profiler=profiler,
            **trainer_params,
        )
