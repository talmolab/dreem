from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.training.losses import AssoLoss
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
import ast
import hydra
import os
import pytorch_lightning as pl
import sleap_io
import sys
import torch
import torch.multiprocessing

# todo: move to config
num_workers = 0
shuffle = True

device = "cuda" if torch.cuda.is_available() else "cpu"

if num_workers > 0:
    # prevent too many open files error
    pin_memory = True
    torch.multiprocessing.set_sharing_strategy("file_system")
else:
    pin_memory = False

# for dataloader if shuffling, since shuffling is done by default on cpu
generator = torch.Generator(device=device) if shuffle == True else None

# useful for longer training runs, but not for single iteration debugging
# finds optimal hardware algs which has upfront time increase for first
# iteration, quicker subsequent iterations

# torch.backends.cudnn.benchmark = True

# pytorch 2 logic - we set our device once here so we don't have to keep setting
torch.set_default_device(device)


class DatasetWrapper(LightningDataModule):
    def __init__(
        self,
        train_ds,
        # val_ds
    ):
        super().__init__()

        self.train_ds = train_ds
        # self.val_ds = val_ds

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=self.train_ds.no_batching_fn,
            num_workers=num_workers,
            generator=generator,
        )

    # def val_dataloader(self):
    # todo: implement val dataloader
    # return DataLoader()


class GTRTrainer(LightningModule):
    def __init__(self, model, loss):
        super().__init__()

        self.model = model
        self.loss = loss

    def training_step(self, train_batch, batch_idx):
        # todo: add logic for wandb logging

        x = train_batch[0]

        logits = self.model(x)

        loss = self.loss(logits, x)

        self.log("train_loss", loss)

        return loss

    # def validation_step(self, val_batch, batch_idx):
    # to implement. also need switch count logic
    # return loss

    def configure_optimizers(self):
        # todo: init from config
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, betas=(0.9, 0.999)
        )

        return optimizer


# not sure we need hydra? could just do argparse + omegaconf?
@hydra.main(config_path="configs", config_name=None, version_base=None)
def train(cfg: DictConfig):
    base_cfg = OmegaConf.load(cfg.base_config)

    if "params_config" in cfg:
        # merge configs
        params_config = OmegaConf.load(cfg.params_config)
        cfg = OmegaConf.merge(base_cfg, params_config)
    else:
        # just use base config
        cfg = base_cfg

    # update with extra cli args
    for arg in sys.argv[1:]:
        if arg.startswith("+"):
            key, val = arg[1:].split("=")
            if key in ["base_config", "params_config"]:
                continue
            try:
                val = ast.literal_eval(val)
            except (SyntaxError, ValueError) as e:
                print(e)
                pass

            OmegaConf.update(cfg, key, val)

    model_params = cfg.model
    dataset_params = cfg.dataset

    gtr_dataset = DatasetWrapper(SleapDataset(**dataset_params))

    gtr_trainer = GTRTrainer(GlobalTrackingTransformer(**model_params), AssoLoss())

    accelerator = "cpu" if device == "cpu" else "gpu"

    # test with 1 epoch and single batch, this should be controlled from config
    trainer = pl.Trainer(max_epochs=1, accelerator=accelerator, limit_train_batches=1)
    trainer.fit(gtr_trainer, gtr_dataset)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py +base_config=configs/base.yaml
    # override with params config:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml
    # override with params config, and specific params:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml +model.norm=True +model.decoder_self_attn=True +dataset.padding=10

    train()
