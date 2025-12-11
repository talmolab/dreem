"""Module containing training, validation and inference logic."""

import gc
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from math import inf
import h5py
import numpy as np
import sleap_io as sio
import tifffile
import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from dreem.datasets import CellTrackingDataset
from dreem.inference import metrics
from dreem.models import GlobalTrackingTransformer
from dreem.models.model_utils import init_optimizer, init_scheduler
from dreem.training.losses import AssoLoss
from dreem.io.flags import FrameFlagCode
from sleap_io.model.suggestions import SuggestionFrame

if TYPE_CHECKING:
    from dreem.io import AssociationMatrix, Frame, Instance

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
        "test": ["motmetrics"],
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
        test_save_path: str = "./test_results.h5",
        save_frame_meta: bool = False,
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
            test_save_path: path to a directory to save the eval and tracking results to
            save_frame_meta: Whether to save frame metadata to the h5 file
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg if model_cfg else {}
        self.loss_cfg = loss_cfg if loss_cfg else {}
        self.tracker_cfg = tracker_cfg if tracker_cfg else {}

        self.model = GlobalTrackingTransformer(**self.model_cfg)
        self.loss = AssoLoss(**self.loss_cfg)
        from dreem.inference.tracker import Tracker

        self.tracker = Tracker(**self.tracker_cfg)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.metrics = metrics if metrics is not None else self.DEFAULT_METRICS
        self.test_results = {
            "preds": [],
            "save_path": test_save_path,
            "save_frame_meta": save_frame_meta,
        }

    def forward(
        self,
        ref_instances: list["Instance"],
        query_instances: list["Instance"] | None = None,
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
        self, train_batch: list[list["Frame"]], batch_idx: int
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
        self, val_batch: list[list["Frame"]], batch_idx: int
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
        self, test_batch: list[list["Frame"]], batch_idx: int
    ) -> dict[str, float]:
        """Execute single test step for model. Performs eval in addition to tracking.

        Args:
            test_batch: A single batch from the dataset which is a list of `Frame` objects
                        with length `clip_length` containing Instances and other metadata.
            batch_idx: the batch number used by lightning

        Returns:
            None
        """
        frames_pred = self.tracker(self.model, test_batch[0])
        self.test_results["preds"].extend([frame.to("cpu") for frame in frames_pred])
        self.log_metrics(None, len(test_batch[0]), "test")
        return None

    def predict_step(self, batch: list[list["Frame"]], batch_idx: int) -> list["Frame"]:
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
        for frame in frames_pred:
            frame = frame.to("cpu")
        return frames_pred

    def _shared_eval_step(self, frames: list["Frame"], mode: str) -> dict[str, float]:
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

            logits = self(instances)
            logits = [asso.matrix for asso in logits]
            loss = self.loss(logits, frames)
            return_metrics = {"loss": loss}
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

    def setup_tracking(self, tracker_cfg: DictConfig, mode: str = "inference"):
        """Setup the tracker for tracking.

        Args:
            tracker_cfg: The configuration for the tracker.
            mode: The mode to run the tracker in.

        Returns:
            A dictionary of overrides for the tracker.
        """
        from dreem.inference.tracker import Tracker

        save_frame_meta = tracker_cfg.cfg.get("save_frame_meta", False)
        self.tracker_cfg = tracker_cfg.get_tracker_cfg()
        self.tracker_cfg["enable_crop_saving"] = (
            save_frame_meta  # to save frame metadata, need to disable crop=None in GTR (memory saving)
        )
        self.tracker = Tracker(**self.tracker_cfg)
        logger.info("Using the following tracker:")
        logger.info(self.tracker)
        if mode == "eval":
            self.metrics["test"] = tracker_cfg.get("metrics", {}).get("test", "all")
            logger.info("Computing the following metrics:")
            logger.info(self.metrics["test"])
            self.test_results["save_frame_meta"] = save_frame_meta
            self.test_results["save_path"] = tracker_cfg.get("outdir", ".")
            os.makedirs(self.test_results["save_path"], exist_ok=True)
        overrides_dict = {
            "max_tracks": self.tracker_cfg.get("max_tracks", inf),
            "save_frame_meta": save_frame_meta,
        }
        return overrides_dict

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
            metrics_to_compute = ["motmetrics"]
        if isinstance(metrics_to_compute, str):
            metrics_to_compute = [metrics_to_compute]
        for metric in metrics_to_compute:
            if metric not in ["motmetrics"]:
                raise ValueError(
                    f"Metric {metric} not supported. Please select from 'motmetrics'"
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
                    frame_switch_map = value[1]
                    motevents = value[2]
                    motevents.to_csv(
                        os.path.join(
                            self.test_results["save_path"], f"{vid_name}.motevents.csv"
                        ),
                        index=False,
                    )
                    mot_summary_group = vid_group.require_group("mot_summary")
                    # Loop through each row in mot_summary and save as attributes
                    for _, row in mot_summary.iterrows():
                        mot_summary_group.attrs[row.name] = row["acc"]
                    if self.test_results["save_frame_meta"]:
                        # save frame metadata for every frame, specifically assoc matrices
                        frame_meta_group = vid_group.require_group("frame_meta")
                        switch_group = frame_meta_group.require_group("switches")
                        for frame in preds:
                            frame = frame.to("cpu")
                            _ = frame.to_h5(
                                frame_meta_group, frame.get_gt_track_ids().cpu().numpy()
                            )
                            if frame.frame_id.item() in frame_switch_map:
                                if frame_switch_map[frame.frame_id.item()]:
                                    switch_group.attrs[
                                        "frame_" + str(frame.frame_id.item())
                                    ] = True
                                else:
                                    switch_group.attrs[
                                        "frame_" + str(frame.frame_id.item())
                                    ] = False

                elif metric_name == "global_tracking_accuracy":
                    gta_by_gt_track = value
                    gta_group = vid_group.require_group("global_tracking_accuracy")
                    # save as a key value pair with gt track id: gta
                    for gt_track_id, gta in gta_by_gt_track.items():
                        gta_group.attrs[f"track_{gt_track_id}"] = gta

        # save the tracking results to a slp/labelled masks file
        if isinstance(self.trainer.test_dataloaders.dataset, CellTrackingDataset):
            outpath = os.path.join(
                self.test_results["save_path"],
                f"{vid_name}.dreem_inference.{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.tif",
            )
            pred_imgs = []
            for frame in preds:
                frame_masks = []
                for instance in frame.instances:
                    # centroid = instance.centroid["centroid"]  # Currently unused but available if needed
                    mask = instance.mask.cpu().numpy()
                    track_id = instance.pred_track_id.cpu().numpy().item()
                    mask = mask.astype(np.uint8)
                    mask[mask != 0] = track_id  # label the mask with the track id
                    frame_masks.append(mask)
                frame_mask = np.max(frame_masks, axis=0)
                pred_imgs.append(frame_mask)
            pred_imgs = np.stack(pred_imgs)
            tifffile.imwrite(outpath, pred_imgs.astype(np.uint16))
        else:
            outpath = os.path.join(
                self.test_results["save_path"],
                f"{vid_name}.dreem_inference.{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.slp",
            )
            pred_slp = []
            suggestions = []
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
                if frame.has_flag(FrameFlagCode.LOW_CONFIDENCE):
                    suggestion = SuggestionFrame(
                        video=video, frame_idx=frame.frame_id.item()
                    )
                    suggestions.append(suggestion)
                lf, tracks = frame.to_slp(tracks, video=video)
                pred_slp.append(lf)
            pred_slp = sio.Labels(pred_slp, suggestions=suggestions)

            pred_slp.save(outpath)

        # clear the preds
        self.test_results["preds"] = []
