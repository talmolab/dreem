"""Script to run inference and get out tracks."""

import logging
import os
from datetime import datetime
from pathlib import Path
import h5py
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sleap_io as sio
from sleap_io.model.suggestions import SuggestionFrame
import tifffile
import torch
from math import inf
from omegaconf import DictConfig
from tqdm import tqdm
from dreem.datasets import CellTrackingDataset
from dreem.inference.tracker import Tracker
from dreem.io import Config, Frame
from dreem.io.flags import FrameFlagCode
from dreem.models import GTRRunner

logger = logging.getLogger("dreem.inference")


def store_frame_metadata(frame, h5_path: str):
    with h5py.File(h5_path, "a") as h5f:
        frame_meta_group = h5f.require_group("frame_meta")
        frame = frame.to("cpu")
        _ = frame.to_h5(
            frame_meta_group,
            frame.get_gt_track_ids().cpu().numpy(),
            save={"features": True, "crop": True},
        )


def get_timestamp() -> str:
    """Get current timestamp.

    Returns:
        the current timestamp in /m/d/y-H:M:S format
    """
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    return date_time


def export_trajectories(
    frames_pred: list[Frame], save_path: str | None = None
) -> pd.DataFrame:
    """Convert trajectories to data frame and save as .csv.

    Args:
        frames_pred: A list of Frames with predicted track ids.
        save_path: The path to save the predicted trajectories to.

    Returns:
        A dictionary containing the predicted track id and centroid coordinates for each instance in the video.
    """
    save_dict = {}
    frame_ids = []
    X, Y = [], []
    pred_track_ids = []
    track_scores = []
    for frame in frames_pred:
        for i, instance in enumerate(frame.instances):
            frame_ids.append(frame.frame_id.item())
            bbox = instance.bbox.squeeze()
            y = (bbox[2] + bbox[0]) / 2
            x = (bbox[3] + bbox[1]) / 2
            X.append(x.item())
            Y.append(y.item())
            track_scores.append(instance.track_score)
            pred_track_ids.append(instance.pred_track_id.item())

    save_dict["Frame"] = frame_ids
    save_dict["X"] = X
    save_dict["Y"] = Y
    save_dict["Pred_track_id"] = pred_track_ids
    save_dict["Track_score"] = track_scores
    save_df = pd.DataFrame(save_dict)
    if save_path:
        save_df.to_csv(save_path, index=False)
    return save_df


def track_ctc(
    model: GTRRunner, trainer: pl.Trainer, dataloader: torch.utils.data.DataLoader
) -> list[pd.DataFrame]:
    """Run Inference.

    Args:
        model: GTRRunner model loaded from checkpoint used for inference
        trainer: lighting Trainer object used for handling inference log.
        dataloader: dataloader containing inference data
    """
    preds = trainer.predict(model, dataloader)
    pred_imgs = []
    for batch in preds:
        for frame in batch:
            frame_masks = []
            for instance in frame.instances:
                # centroid = instance.centroid["centroid"]  # Currently unused but available if needed
                mask = instance.mask.cpu().numpy()
                track_id = instance.pred_track_id.cpu().numpy().item()
                mask = mask.astype(np.uint8)
                mask[mask != 0] = track_id  # label the mask with the track id
                frame_masks.append(mask)
                # combine masks by merging into single mask using max; this will cause overlap
            frame_mask = np.max(frame_masks, axis=0)
            pred_imgs.append(frame_mask)
    pred_imgs = np.stack(pred_imgs)
    return pred_imgs


def track(
    model: GTRRunner,
    trainer: pl.Trainer,
    dataloader: torch.utils.data.DataLoader,
    outdir: str,
    overrides_dict: dict,
) -> list[pd.DataFrame]:
    """Run Inference.

    Args:
        model: GTRRunner model loaded from checkpoint used for inference
        trainer: lighting Trainer object used for handling inference log.
        dataloader: dataloader containing inference data

    Return:
        List of DataFrames containing prediction results for each video
    """
    suggestions = []
    preds = trainer.predict(model, dataloader)
    save_frame_meta = overrides_dict["save_frame_meta"]
    if save_frame_meta:
        h5_path = os.path.join(
            outdir,
            f"{dataloader.dataset.slp_files[0].split('/')[-1].replace('.slp', '')}_frame_meta.h5",
        )
        if os.path.exists(h5_path):
            os.remove(h5_path)
        with h5py.File(h5_path, "a") as h5f:
            h5f.create_dataset("vid_name", data=preds[0][0].vid_name)
    pred_slp = []
    tracks = {}
    for batch in tqdm(preds, desc="Saving .slp and frame metadata"):
        for frame in batch:
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
            if save_frame_meta:
                store_frame_metadata(frame, h5_path)
    pred_slp = sio.Labels(pred_slp, suggestions=suggestions)
    return pred_slp


@hydra.main(config_path=None, config_name=None, version_base=None)
def run(cfg: DictConfig) -> dict[int, sio.Labels]:
    """Run inference based on config file.

    Args:
        cfg: A dictconfig loaded from hydra containing checkpoint path and data
    """
    pred_cfg = Config(cfg)

    if "checkpoints" in cfg.keys():
        try:
            index = int(os.environ["POD_INDEX"])
        # For testing without deploying a job on runai
        except KeyError:
            index = input("Pod Index Not found! Please choose a pod index: ")

        logger.info(f"Pod Index: {index}")

        checkpoints = pd.read_csv(cfg.checkpoints)
        checkpoint = checkpoints.iloc[index]
    else:
        checkpoint = pred_cfg.cfg.ckpt_path

    logging.getLogger().setLevel(level=cfg.get("log_level", "INFO").upper())

    if not checkpoint:
        raise ValueError(
            "Model checkpoint not found. Please provide a valid checkpoint path."
        )

    model = GTRRunner.load_from_checkpoint(checkpoint, strict=False)
    overrides_dict = model.setup_tracking(pred_cfg, mode="inference")

    labels_files, vid_files = pred_cfg.get_data_paths(
        "test", pred_cfg.cfg.dataset.test_dataset
    )
    trainer = pred_cfg.get_trainer()
    outdir = pred_cfg.cfg.outdir if "outdir" in pred_cfg.cfg else "./results"
    os.makedirs(outdir, exist_ok=True)

    for label_file, vid_file in zip(labels_files, vid_files):
        dataset = pred_cfg.get_dataset(
            label_files=[label_file],
            vid_files=[vid_file],
            mode="test",
            overrides=overrides_dict,
        )
        dataloader = pred_cfg.get_dataloader(dataset, mode="test")
        if isinstance(vid_file, list):
            save_file_name = vid_file[0].split("/")[-2]
        else:
            save_file_name = vid_file

        if isinstance(dataset, CellTrackingDataset):
            preds = track_ctc(model, trainer, dataloader)
            outpath = os.path.join(
                outdir,
                f"{Path(save_file_name).stem}.dreem_inference.{get_timestamp()}.tif",
            )
            tifffile.imwrite(outpath, preds.astype(np.uint16))
        else:
            preds = track(model, trainer, dataloader, outdir, overrides_dict)
            outpath = os.path.join(
                outdir,
                f"{Path(save_file_name).stem}.dreem_inference.{get_timestamp()}.slp",
            )
            preds.save(outpath)

    return preds


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py --config-dir=./configs --config-name=inference

    # override with params config:
    # python train.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python train.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml dataset.train_dataset.padding=10
    run()
