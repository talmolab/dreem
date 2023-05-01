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
generator = torch.Generator(device=device) if shuffle else None

# useful for longer training runs, but not for single iteration debugging
# finds optimal hardware algs which has upfront time increase for first
# iteration, quicker subsequent iterations

# torch.backends.cudnn.benchmark = True

# pytorch 2 logic - we set our device once here so we don't have to keep setting
torch.set_default_device(device)


def train(model, dataset, trainer):
    trainer.fit(model, dataset)


# not sure we need hydra? could just do argparse + omegaconf?
@hydra.main(config_path="configs", config_name=None, version_base=None)
def main(cfg: DictConfig):
    cfg = Config(cfg)

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

    cfg.update(hparams)

    model = cfg.get_model()
    dataset = cfg.get_dataset()
    loss = cfg.get_loss()

    dataset = TrackingDataset(dataset)

    model = GTRRunner(model, loss)

    accelerator = "cpu" if device == "cpu" else "gpu"

    # test with 1 epoch and single batch, this should be controlled from config
    trainer = pl.Trainer(max_epochs=1, accelerator=accelerator, limit_train_batches=1)

    train(model, dataset, trainer)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py +base_config=configs/base.yaml
    # override with params config:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml
    # override with params config, and specific params:
    # python train.py +base_config=configs/base.yaml +params_config=configs/params.yaml +model.norm=True +model.decoder_self_attn=True +dataset.padding=10
    main()
