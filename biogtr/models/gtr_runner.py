from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from biogtr.inference.tracker import Tracker
from biogtr.inference import metrics
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
        train_metrics=[""],
        val_metrics=["sw_cnt"],
    ):
        """Initialize a lightning module for GTR
        Args:
            model: GlobalTrackingTransformer model to be trained/used for eval
            tracker_cfg: The parameters used for the tracker post-processing
            loss: AssoLoss function to optimize
            optimizer: optimizer to train model with. Only used to overwrite `configure_optimizer`
            scheduler: lr_scheduler used to overwrite `configure_optimizer
            train_metrics: a list of metrics to be calculated during training outside of loss
            val_metrics: a list of metrics to be calculated during validation outside of loss
        """
        super().__init__()

        self.model = model
        self.tracker_cfg = tracker_cfg
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    def forward(self, instances) -> torch.Tensor:
        """The forward pass of the lightning module
        Args:
            instances: a list of dicts where each dict is a frame with gt data
        Returns: An association matrix between objects
        """
        return self.model(instances)

    def training_step(
        self, train_batch: list[dict], batch_idx: int
    ) -> dict[str, float]:
        """Method outlining the training procedure for model
        Args:
            train_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning
        Returns: A dict containing the train loss plus any other metrics specified during initialization
        """
        result = self._shared_eval_step(train_batch[0], self.train_metrics)
        for metric, val in result.items():
            self.log(f"train_{metric}", val, batch_size=len(train_batch[0]))
        return result

    def validation_step(
        self, val_batch: list[dict], batch_idx: int
    ) -> dict[str, float]:
        """Method outlining the training procedure for model
        Args:
            val_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning
        Returns: A dict containing the val loss plus any other metrics specified during initialization
        """
        result = self._shared_eval_step(val_batch[0], eval_metrics=self.val_metrics)
        for metric, val in result.items():
            self.log(f"val_{metric}", val, batch_size=len(val_batch[0]))
        return result

    def _shared_eval_step(self, instances, eval_metrics=["sw_cnt"]):
        """Helper function for running evaluation used by train, test, and val steps
        Args:
            instances: A list of dicts where each dict is a frame containing gt data
            eval_metrics: A list of metrics calculated and saved
        Returns: a dict containing the loss and any other metrics specified by `eval_metrics`
        """
        logits = self.model(instances)
        loss = self.loss(logits, instances)

        return_metrics = {"loss": loss}
        if "sw_cnt" in eval_metrics:
            tracker = Tracker(self.model, **self.tracker_cfg)
            instances_pred = tracker.track(instances)
            matches, indices, _ = metrics.get_matches(instances_pred)
            switches = metrics.get_switches(matches, indices)
            sw_cnt = metrics.get_switch_count(switches)
            return_metrics["sw_cnt"] = sw_cnt
        return return_metrics

    def configure_optimizers(self) -> dict:
        """
        Get optimizers and schedulers for training. Is overridden by config but defaults to Adam + ReduceLROnPlateau
        Returns: an optimizer config dict containing the optimizer, scheduler, and scheduler params
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
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 10,
            },
        }
