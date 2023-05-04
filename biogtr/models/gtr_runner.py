from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.training.losses import AssoLoss
from pytorch_lightning import LightningModule

"""
Lightning Wrapper around GlobalTrackingTransformer. Used to train model + run eval
"""


class GTRRunner(LightningModule):
    def __init__(
        self,
        model: GlobalTrackingTransformer,
        loss: AssoLoss,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        train_metrics=[""],
    ):
        """
        Initialize a lightning module for GTR
        Args:
            model: GlobalTrackingTransformer model to be trained/used for eval
            loss: AssoLoss function to optimize
            optimizer: optimizer to train model with. Only used to overwrite `configure_optimizer`
            scheduler: lr_scheduler used to overwrite `configure_optimizer
        """
        super().__init__()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = train_metrics

    def forward(self, instances):
        return self.model(instances)

    def training_step(self, train_batch, batch_idx):
        # todo: add logic for wandb logging
        result = self._shared_eval_step(train_batch[0], self.train_metrics)
        for metric, val in result.items():
            self.log(f"train_{metric}", val, batch_size=len(train_batch[0]))
        return result

    def validation_step(self, val_batch, batch_idx):
        result = self._shared_eval_step(val_batch[0])
        for metric, val in result.items():
            self.log(f"train_{metric}", val, batch_size=len(val_batch[0]))
        return result

    def _shared_eval_step(self, instances, eval_metrics=["sw_cnt"]):
        logits = self.model(instances)

        loss = self.loss(logits, instances)

        return_metrics = {"loss": loss}

        return return_metrics

    # def validation_step(self, val_batch, batch_idx):
    # to implement. also need switch count logic
    # return loss

    def configure_optimizers(self):
        # todo: init from config
        if self.optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-4, betas=(0.9, 0.999)
            )
        else:
            optimizer = self.optimizer

        if self.scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", 0.5, 10
            )
        else:
            scheduler = self.scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 10,
            },
        }
