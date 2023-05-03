from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from biogtr.inference import metrics
from biogtr.inference.tracker import Tracker
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
        tracker_cfg: dict,
        loss: AssoLoss,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        train_metrics: list[str] = [],
        val_metrics: list[str] = ["switch_count"],
        test_metrics: list[str] = ["switch_count"],
    ):
        """
        Initialize a lightning module for GTR
        Args:
            model: GlobalTrackingTransformer model to be trained/used for eval
            tracker_cfg: post_processing params to be used with tracker
            loss: AssoLoss function to optimize
            optimizer: optimizer to train model with. Only used to overwrite `configure_optimizer`
            scheduler: lr_scheduler used to overwrite `configure_optimizer`
            train_metrics: Metrics to be calculated during train step
            val_metrics: Metrics to be calculated during val step
            test_metrics: Metrics to be calculated during test step
        """
        super().__init__()

        self.model = model
        self.tracker_cfg = tracker_cfg
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def forward(self, x: list[dict]):
        """
        Forward pass of model. Output association matrix
        Returns: The association matrix without any postprocessing
        Args: x: a list of dicts where each dict is a frame and contains gt data
        """
        logits = self.model(x)
        return logits

    def training_step(self, train_batch: list[dict], batch_idx) -> dict:
        """
        Train step of model.
        Returns: a dictionary containing the training metrics calculated at each step
        Args:
            train_batch: a list of dicts where each dict is a frame and contains gt data
        """
        print(type(train_batch[0]))
        eval_metrics = self._shared_eval_step(
            train_batch, eval_metrics=self.train_metrics
        )

        for metric, val in eval_metrics:
            self.log(f"train_{metric}", val)

        return eval_metrics

    def validation_step(self, val_batch: list[dict], batch_idx) -> dict:
        """
        Validation step of model.
        Returns: a dictionary containing the validation metrics calculated at each step
        Args:
            test_batch: a list of dicts where each dict is a frame and contains gt data
        """
        eval_metrics = self._shared_eval_step(val_batch, eval_metrics=self.val_metrics)

        for metric, val in eval_metrics:
            self.log(f"val_{metric}", val)

        return eval_metrics

    def test_step(self, test_batch, batch_idx):
        """
        Train step of model.
        Returns: a dictionary containing the test metrics calculated at each step
        Args:
            test_batch: a list of dicts where each dict is a frame and contains gt data
        """
        eval_metrics = self._shared_eval_step(
            test_batch, eval_metrics=self.test_metrics
        )

        for metric, val in eval_metrics:
            self.log(f"test_{metric}", val)

        return eval_metrics

    def predict_step(self, batch: list[dict]) -> list[dict]:
        """
        Run prediction on held out data. Get out
        Returns: a list of dicts where each dict is a frame and contains the predicted track ids
        Args:
            test_batch: a list of dicts where each dict is a frame and contains gt data
        """
        tracker = Tracker(self.model, **self.tracker_cfg)
        instances_pred = tracker(batch)
        return instances_pred

    def _shared_eval_step(self, instances, eval_metrics=["switch_count"]):
        logits = self(instances)
        loss = self.loss(logits, instances)
        return_metrics = {"loss": loss}

        if "switch_count" in eval_metrics:
            tracker = Tracker(self.model, **self.tracker_cfg)
            instances_pred = tracker(instances)
            matches, indices, _ = metrics.get_matches(instances_pred)
            switches = metrics.get_switches(matches, indices)
            sw_cnt = metrics.get_switch_count(switches)
            return_metrics["sw_cnt"] = sw_cnt

        return return_metrics

    def configure_optimizers(self) -> dict:
        """
        configure optimizer and scheduler to be used during training
        """
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
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 10,
            },
        }
