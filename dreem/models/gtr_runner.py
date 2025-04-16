"""Module containing training, validation and inference logic."""

import torch
import gc
import logging
import pandas as pd
import h5py
import os
from dreem.inference import Tracker, BatchTracker
from dreem.inference import metrics
from dreem.models import GlobalTrackingTransformer
from dreem.training.losses import AssoLoss
from dreem.models.model_utils import init_optimizer, init_scheduler
from pytorch_lightning import LightningModule
from datetime import datetime
from pathlib import Path
import sleap_io as sio

logger = logging.getLogger("dreem.models")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)


class GTRRunner(LightningModule):
    """A lightning wrapper around GTR model.

    Used for training, validation and inference.
    """

    DEFAULT_METRICS = {
        "train": [],
        "val": [],
        "test": ["num_switches", "global_tracking_accuracy"],
    }
    DEFAULT_TRACKING = {
        "train": False,
        "val": False,
        "test": True,
    }
    DEFAULT_SAVE = {"train": False, "val": False, "test": False}

    def __init__(
        self,
        model_cfg: dict | None = None,
        tracker_cfg: dict | None = None,
        loss_cfg: dict | None = None,
        optimizer_cfg: dict | None = None,
        scheduler_cfg: dict | None = None,
        metrics: dict[str, list[str]] | None = None,
        persistent_tracking: dict[str, bool] | None = None,
        test_save_path: str = "./test_results.h5",
    ):
        """Initialize a lightning module for GTR.

        Args:
            model_cfg: hyperparameters for GlobalTrackingTransformer
            tracker_cfg: The parameters used for the tracker post-processing
            loss_cfg: hyperparameters for AssoLoss
            optimizer_cfg: hyper parameters used for optimizer.
                       Only used to overwrite `configure_optimizer`
            scheduler_cfg: hyperparameters for lr_scheduler used to overwrite `configure_optimizer
            metrics: a dict containing the metrics to be computed during train, val, and test.
            persistent_tracking: a dict containing whether to use persistent tracking during train, val and test inference.
            test_save_path: path to a directory to save the eval and tracking results to
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg if model_cfg else {}
        self.loss_cfg = loss_cfg if loss_cfg else {}
        self.tracker_cfg = tracker_cfg if tracker_cfg else {}

        self.model = GlobalTrackingTransformer(**self.model_cfg)
        self.loss = AssoLoss(**self.loss_cfg)
        if self.tracker_cfg.get("tracker_type", "standard") == "batch":
            self.tracker = BatchTracker(**self.tracker_cfg)
        else:
            self.tracker = Tracker(**self.tracker_cfg)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.metrics = metrics if metrics is not None else self.DEFAULT_METRICS
        self.persistent_tracking = (
            persistent_tracking
            if persistent_tracking is not None
            else self.DEFAULT_TRACKING
        )
        self.test_results = {"preds": [], "save_path": test_save_path}

    def forward(
        self,
        ref_instances: list["dreem.io.Instance"],
        query_instances: list["dreem.io.Instance"] | None = None,
    ) -> list["AssociationMatrix"]:
        """Execute forward pass of the lightning module.

        Args:
            ref_instances: a list of `Instance` objects containing crops and other data needed for transformer model
            query_instances: a list of `Instance` objects used as queries in the decoder. Mostly used for inference.

        Returns:
            An association matrix between objects
        """
        asso_preds = self.model(ref_instances, query_instances)
        return asso_preds

    def training_step(
        self, train_batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> dict[str, float]:
        """Execute single training step for model.

        Args:
            train_batch: A single batch from the dataset which is a list of `Frame` objects
                        with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the train loss plus any other metrics specified
        """
        result = self._shared_eval_step(train_batch[0], mode="train")
        self.log_metrics(result, len(train_batch[0]), "train")

        return result

    def validation_step(
        self, val_batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> dict[str, float]:
        """Execute single val step for model.

        Args:
            val_batch: A single batch from the dataset which is a list of `Frame` objects
                        with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(val_batch[0], mode="val")
        self.log_metrics(result, len(val_batch[0]), "val")

        return result

    def test_step(
        self, test_batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> dict[str, float]:
        """Execute single test step for model.

        Args:
            test_batch: A single batch from the dataset which is a list of `Frame` objects
                        with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(test_batch[0], mode="test")
        self.log_metrics(result, len(test_batch[0]), "test")

        return result

    def predict_step(
        self, batch: list[list["dreem.io.Frame"]], batch_idx: int
    ) -> list["dreem.io.Frame"]:
        """Run inference for model.

        Computes association + assignment.

        Args:
            batch: A single batch from the dataset which is a list of `Frame` objects
                    with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            A list of dicts where each dict is a frame containing the predicted track ids
        """
        frames_pred = self.tracker(self.model, batch[0])
        return frames_pred

    def _shared_eval_step(
        self, frames: list["dreem.io.Frame"], mode: str
    ) -> dict[str, float]:
        """Run evaluation used by train, test, and val steps.

        Args:
            frames: A list of dicts where each dict is a frame containing gt data
            mode: which metrics to compute and whether to use persistent tracking or not

        Returns:
            a dict containing the loss and any other metrics specified by `eval_metrics`
        """
        try:
            instances = [instance for frame in frames for instance in frame.instances]

            if len(instances) == 0:
                return None

            eval_metrics = self.metrics[mode]

            logits = self(instances)
            logits = [asso.matrix for asso in logits]
            loss = self.loss(logits, frames)

            return_metrics = {"loss": loss}
            if mode == "test":
                self.tracker.persistent_tracking = True
                frames_pred = self.tracker(self.model, frames)
                self.test_results["preds"].extend(
                    [frame.to("cpu") for frame in frames_pred]
                )
            return_metrics["batch_size"] = len(frames)
        except Exception as e:
            logger.exception(
                f"Failed on frame {frames[0].frame_id} of video {frames[0].video_id}"
            )
            logger.exception(e)
            raise (e)

        return return_metrics

    def configure_optimizers(self) -> dict:
        """Get optimizers and schedulers for training.

        Is overridden by config but defaults to Adam + ReduceLROnPlateau.

        Returns:
            an optimizer config dict containing the optimizer, scheduler, and scheduler params
        """
        # todo: init from config
        if self.optimizer_cfg is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        else:
            optimizer = init_optimizer(self.parameters(), self.optimizer_cfg)

        if self.scheduler_cfg is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", 0.5, 10
            )
        else:
            scheduler = init_scheduler(optimizer, self.scheduler_cfg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def log_metrics(self, result: dict, batch_size: int, mode: str) -> None:
        """Log metrics computed during evaluation.

        Args:
            result: A dict containing metrics to be logged.
            batch_size: the size of the batch used to compute the metrics
            mode: One of {'train', 'test' or 'val'}. Used as prefix while logging.
        """
        if result:
            batch_size = result.pop("batch_size")
            for metric, val in result.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.log(f"{mode}_{metric}", val, batch_size=batch_size)

    def on_validation_epoch_end(self):
        """Execute hook for validation end.

        Currently, we simply clear the gpu cache and do garbage collection.
        """
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_end(self):
        """Run inference and metrics pipeline to compute metrics for test set.

        Args:
            test_results: dict containing predictions and metrics to be filled out in metrics.evaluate
            metrics: list of metrics to compute
        """
        # input validation
        metrics_to_compute = self.metrics[
            "test"
        ]  # list of metrics to compute, or "all"
        if metrics_to_compute == "all":
            metrics_to_compute = ["motmetrics", "global_tracking_accuracy"]
        if isinstance(metrics_to_compute, str):
            metrics_to_compute = [metrics_to_compute]
        for metric in metrics_to_compute:
            if metric not in ["motmetrics", "global_tracking_accuracy"]:
                raise ValueError(
                    f"Metric {metric} not supported. Please select from 'motmetrics' or 'global_tracking_accuracy'"
                )

        preds = self.test_results["preds"]

        # results is a dict with key being the metric name, and value being the metric value computed
        results = metrics.evaluate(preds, metrics_to_compute)

        # save metrics and frame metadata to hdf5

        # Get the video name from the first frame
        vid_name = Path(preds[0].vid_name).stem
        # save the results to an hdf5 file
        fname = os.path.join(
            self.test_results["save_path"], f"{vid_name}.dreem_metrics.h5"
        )
        logger.info(f"Saving metrics to {fname}")
        # Check if the h5 file exists and add a suffix to prevent name collision
        suffix_counter = 0
        original_fname = fname
        while os.path.exists(fname):
            suffix_counter += 1
            fname = original_fname.replace(
                ".dreem_metrics.h5", f"_{suffix_counter}.dreem_metrics.h5"
            )

        if suffix_counter > 0:
            logger.info(f"File already exists. Saving to {fname} instead")

        with h5py.File(fname, "a") as results_file:
            # Create a group for this video
            vid_group = results_file.require_group(vid_name)
            # Save each metric
            for metric_name, value in results.items():
                if metric_name == "motmetrics":
                    # For num_switches, save mot_summary and mot_events separately
                    mot_summary = value[0]
                    mot_events = value[1]
                    frame_switch_map = value[2]
                    mot_summary_group = vid_group.require_group("mot_summary")
                    # Loop through each row in mot_summary and save as attributes
                    for _, row in mot_summary.iterrows():
                        mot_summary_group.attrs[row.name] = row["acc"]
                    # save extra metadata for frames in which there is a switch
                    for frame_id, switch in frame_switch_map.items():
                        frame = preds[frame_id]
                        frame = frame.to("cpu")
                        if switch:
                            _ = frame.to_h5(
                                vid_group,
                                frame.get_gt_track_ids().cpu().numpy(),
                                save={
                                    "crop": True,
                                    "features": True,
                                    "embeddings": True,
                                },
                            )
                        else:
                            _ = frame.to_h5(
                                vid_group, frame.get_gt_track_ids().cpu().numpy()
                            )
                    # save motevents log to csv
                    motevents_path = os.path.join(
                        self.test_results["save_path"], f"{vid_name}.motevents.csv"
                    )
                    logger.info(f"Saving motevents log to {motevents_path}")
                    mot_events.to_csv(motevents_path, index=False)

                elif metric_name == "global_tracking_accuracy":
                    gta_by_gt_track = value
                    gta_group = vid_group.require_group("global_tracking_accuracy")
                    # save as a key value pair with gt track id: gta
                    for gt_track_id, gta in gta_by_gt_track.items():
                        gta_group.attrs[f"track_{gt_track_id}"] = gta

        # save the tracking results to a slp file
        pred_slp = []
        outpath = os.path.join(
            self.test_results["save_path"],
            f"{vid_name}.dreem_inference.{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.slp",
        )
        logger.info(f"Saving inference results to {outpath}")
        # save the tracking results to a slp file
        tracks = {}
        for frame in preds:
            if frame.frame_id.item() == 0:
                video = (
                    sio.Video(frame.video)
                    if isinstance(frame.video, str)
                    else sio.Video
                )
            lf, tracks = frame.to_slp(tracks, video=video)
            pred_slp.append(lf)
        pred_slp = sio.Labels(pred_slp)

        pred_slp.save(outpath)
        # clear the preds
        self.test_results["preds"] = []
