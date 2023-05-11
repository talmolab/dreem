"""Training script for training model.

Used for training a single model or deploying a batch train job on RUNAI CLI
"""
from biogtr.config import Config
from biogtr.datasets.tracking_dataset import TrackingDataset
from biogtr.models.gtr_runner import GTRRunner
from omegaconf import DictConfig
from pprint import pprint

import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import sys
import torch
import torch.multiprocessing

device = "cuda" if torch.cuda.is_available() else "cpu"

# useful for longer training runs, but not for single iteration debugging
# finds optimal hardware algs which has upfront time increase for first
# iteration, quicker subsequent iterations

# torch.backends.cudnn.benchmark = True

# pytorch 2 logic - we set our device once here so we don't have to keep setting
torch.set_default_device(device)


# not sure we need hydra? could just do argparse + omegaconf?
@hydra.main(config_path="configs", config_name=None, version_base=None)
def main(cfg: DictConfig):
    """Main function for training.

    Handles all config parsing and initialization then calls `trainer.train()`.
    If `batch_config` is included then run will be assumed to be a batch job.

    Args:
        cfg: The config dict parsed by `hydra`
    """
    train_cfg = Config(cfg)
    pprint(f"Base train config: {train_cfg.cfg}")

    # update with parameters for batch train job
    if "batch_config" in cfg.keys():
        try:
            index = int(os.environ["POD_INDEX"])
        # For testing without deploying a job on runai
        except KeyError:
            print("Pod Index Not found! Setting index to 0")
            index = 0
        print(f"Pod Index: {index}")

        hparams_df = pd.read_csv(cfg.batch_config)
        hparams = hparams_df.iloc[index].to_dict()

        print("Updating the following hparams to the following values")
        pprint(hparams)
        train_cfg.set_hparams(hparams)
    print(f"Updated train config: {train_cfg}")
    # update with extra cli args
    hparams_cli = {}
    for arg in sys.argv[1:]:
        if arg.startswith("+"):
            key, val = arg[1:].split("=")
            if key in ["base_config", "params_config"]:
                continue
            try:
                hparams_cli[key] = val
            except (SyntaxError, ValueError) as e:
                print(e)
                pass

    train_cfg.set_hparams(hparams_cli)

    model = train_cfg.get_model()
    train_dataset = train_cfg.get_dataset(type="sleap", mode="train")
    train_dataloader = train_cfg.get_dataloader(train_dataset, mode="train")

    val_dataset = train_cfg.get_dataset(type="sleap", mode="val")
    val_dataloader = train_cfg.get_dataloader(val_dataset, mode="val")

    test_dataset = train_cfg.get_dataset(type="sleap", mode="test")
    test_dataloader = train_cfg.get_dataloader(test_dataset, mode="test")

    dataset = TrackingDataset(
        train_dl=train_dataloader, val_dl=val_dataloader, test_dl=test_dataloader
    )

    model = train_cfg.get_gtr_runner()

    # test with 1 epoch and single batch, this should be controlled from config
    # todo: get to work with multi-gpu training
    logger = train_cfg.get_logger()

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        train_cfg.get_checkpointing("./models"),
        train_cfg.get_early_stopping(),
    ]

    trainer = train_cfg.get_trainer(
        callbacks,
        logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "cpu",
    )

    ckpt_path = train_cfg.get_ckpt_path()
    trainer.fit(model, dataset, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py +base_config=configs/base.yaml
    # override with params config:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml
    # override with params config, and specific params:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml +model.norm=True +model.decoder_self_attn=True +dataset.padding=10
    # deploy batch train job:
    # python train.py +base_config=configs/base.yaml +batch_config=test_batch_train.yaml
    main()
