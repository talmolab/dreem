from biogtr.config import Config
from biogtr.datasets.tracking_dataset import TrackingDataset
from biogtr.models.gtr_runner import GTRRunner
from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl
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
generator = torch.Generator(device=device) if shuffle and device == "cuda" else None

# useful for longer training runs, but not for single iteration debugging
# finds optimal hardware algs which has upfront time increase for first
# iteration, quicker subsequent iterations

# torch.backends.cudnn.benchmark = True

# pytorch 2 logic - we set our device once here so we don't have to keep setting
torch.set_default_device(device)


# not sure we need hydra? could just do argparse + omegaconf?
@hydra.main(config_path="configs", config_name=None, version_base=None)
def train(cfg: DictConfig):
    train_cfg = Config(cfg)

    # update with extra cli args
    hparams = {}
    for arg in sys.argv[1:]:
        if arg.startswith("+"):
            key, val = arg[1:].split("=")
            if key in ["base_config", "params_config"]:
                continue
            try:
                hparams[key] = val
            except (SyntaxError, ValueError) as e:
                print(e)
                pass

    train_cfg.set_hparams(hparams)

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

    loss = train_cfg.get_loss()

    optimizer = train_cfg.get_optimizer(model.parameters())
    scheduler = train_cfg.get_scheduler(optimizer)

    tracker_cfg = train_cfg.get_tracker_cfg()
    model = GTRRunner(
        model=model,
        tracker_cfg=tracker_cfg,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # test with 1 epoch and single batch, this should be controlled from config
    # todo: get to work with multi-gpu training
    logger = train_cfg.get_logger()
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        train_cfg.get_early_stopping(),
        train_cfg.get_checkpointing("./models"),
    ]
    trainer = train_cfg.get_trainer(callbacks, logger)
    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py +base_config=configs/base.yaml
    # override with params config:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml
    # override with params config, and specific params:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml +model.norm=True +model.decoder_self_attn=True +dataset.padding=10
    train()
