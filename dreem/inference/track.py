"""API for running tracking inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import sleap_io as sio
import tifffile
import torch
from omegaconf import DictConfig
from sleap_io.model.suggestions import SuggestionFrame
from tqdm import tqdm

from dreem.datasets import CellTrackingDataset
from dreem.io import Config, Frame
from dreem.io.flags import FrameFlagCode
from dreem.io.pretrained import resolve_checkpoint
from dreem.models import GTRRunner

logger = logging.getLogger("dreem.inference")


def store_frame_metadata(frame, h5_path: str):
    """Store frame metadata to HDF5 file."""
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
        the current timestamp in m-d-Y-H-M-S format
    """
    return datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def export_trajectories(
    frames_pred: list[Frame], save_path: str | None = None
) -> pd.DataFrame:
    """Convert tracked frames to a tabular DataFrame and optionally save as CSV.

    Args:
        frames_pred: A list of Frames with predicted track ids.
        save_path: The path to save the predicted trajectories to.

    Returns:
        A DataFrame with columns: frame, detection_idx, track_id, confidence,
        centroid_x, centroid_y.
    """
    import pandas as pd

    rows: list[dict] = []
    for frame in frames_pred:
        frame_id = frame.frame_id.item()
        for det_idx, instance in enumerate(frame.instances):
            centroid = instance.centroid.get("centroid")
            if centroid is not None:
                cx, cy = float(centroid[0]), float(centroid[1])
            else:
                bbox = instance.bbox.squeeze()
                cx = float((bbox[3] + bbox[1]) / 2)
                cy = float((bbox[2] + bbox[0]) / 2)
            rows.append(
                {
                    "frame": frame_id,
                    "detection_idx": det_idx,
                    "track_id": instance.pred_track_id.item(),
                    "confidence": instance.track_score,
                    "centroid_x": cx,
                    "centroid_y": cy,
                }
            )
    columns = [
        "frame",
        "detection_idx",
        "track_id",
        "confidence",
        "centroid_x",
        "centroid_y",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def track_ctc(
    model: GTRRunner, trainer: pl.Trainer, dataloader: torch.utils.data.DataLoader
) -> tuple[np.ndarray, list[Frame]]:
    """Run tracking inference for Cell Tracking Challenge format.

    Args:
        model: GTRRunner model loaded from checkpoint used for inference
        trainer: Lightning Trainer object used for handling inference.
        dataloader: Dataloader containing inference data

    Returns:
        Tuple of (stacked numpy array of predicted mask images, list of Frame
        objects with predicted track ids).
    """
    preds = trainer.predict(model, dataloader)
    pred_imgs = []
    all_frames: list[Frame] = []
    for batch in preds:
        for frame in batch:
            all_frames.append(frame)
            frame_masks = []
            for instance in frame.instances:
                mask = instance.mask.cpu().numpy()
                track_id = instance.pred_track_id.cpu().numpy().item()
                mask = mask.astype(np.uint16)
                mask[mask != 0] = track_id
                frame_masks.append(mask)

            if frame_masks:
                frame_mask = np.max(frame_masks, axis=0)
            else:
                # Handle empty instances case
                img_shape = frame.img_shape
                if len(img_shape) == 3:
                    _, height, width = img_shape
                elif len(img_shape) == 2:
                    height, width = img_shape
                frame_mask = np.zeros((height, width), dtype=np.uint16)

            pred_imgs.append(frame_mask)
    pred_imgs = np.stack(pred_imgs)
    return pred_imgs, all_frames


def track_sleap(
    model: GTRRunner,
    trainer: pl.Trainer,
    dataloader: torch.utils.data.DataLoader,
    outdir: str,
    overrides_dict: dict,
) -> tuple[sio.Labels, list[Frame]]:
    """Run tracking inference for SLEAP format.

    Args:
        model: GTRRunner model loaded from checkpoint used for inference
        trainer: Lightning Trainer object used for handling inference.
        dataloader: Dataloader containing inference data
        outdir: Output directory for saving results
        overrides_dict: Dictionary of config overrides

    Returns:
        Tuple of (SLEAP Labels object with predicted tracks, list of Frame
        objects with predicted track ids).
    """
    suggestions = []
    preds = trainer.predict(model, dataloader)
    save_frame_meta = overrides_dict.get("save_frame_meta", False)
    if save_frame_meta:
        h5_path = os.path.join(
            outdir,
            f"{Path(dataloader.dataset.slp_files[0]).stem}_frame_meta.h5",
        )
        if os.path.exists(h5_path):
            os.remove(h5_path)
        with h5py.File(h5_path, "a") as h5f:
            h5f.create_dataset("vid_name", data=preds[0][0].vid_name)
    pred_slp = []
    tracks = {}
    all_frames: list[Frame] = []
    for batch in tqdm(preds, desc="Saving .slp and frame metadata"):
        for frame in batch:
            all_frames.append(frame)
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
    return pred_slp, all_frames


def _summarize_preds(
    preds: sio.Labels | np.ndarray,
) -> dict:
    """Extract summary statistics from predictions.

    Args:
        preds: Predictions (SLEAP Labels or numpy array for CTC)

    Returns:
        Dict with num_frames, num_tracks, and track_ids (as Python ints)
    """
    if isinstance(preds, np.ndarray):
        track_ids = sorted(int(x) for x in set(np.unique(preds)) - {0})
        num_frames = preds.shape[0]
    else:
        all_ids = set()
        num_frames = len(preds)
        for lf in preds:
            for inst in lf.instances:
                tid = inst.track.name if inst.track else None
                if tid is not None:
                    all_ids.add(int(tid))
        track_ids = sorted(all_ids)
    return {
        "num_frames": num_frames,
        "num_tracks": len(track_ids),
        "track_ids": track_ids,
    }


def run(cfg: DictConfig) -> dict:
    """Run tracking inference based on config.

    Args:
        cfg: A DictConfig containing checkpoint path and data configuration

    Returns:
        Dict with keys: "preds" (SLEAP Labels or numpy array), "output_paths"
        (list of absolute path strings), and "summary" (dict with num_frames,
        num_tracks, track_ids as Python ints). Returns empty results if no
        data found.
    """
    pred_cfg = Config(cfg)

    checkpoint = pred_cfg.cfg.ckpt_path
    if not checkpoint:
        raise ValueError(
            "Model checkpoint not found. Please provide a valid checkpoint path."
        )

    checkpoint = resolve_checkpoint(checkpoint)
    model = GTRRunner.load_from_checkpoint(checkpoint, strict=False)
    overrides_dict = model.setup_tracking(pred_cfg, mode="inference")

    labels_files, vid_files = pred_cfg.get_data_paths(
        "test", pred_cfg.cfg.dataset.test_dataset
    )
    trainer = pred_cfg.get_trainer(mode="inference")
    outdir = pred_cfg.cfg.outdir if "outdir" in pred_cfg.cfg else "./results"
    os.makedirs(outdir, exist_ok=True)

    output_format = pred_cfg.cfg.get("output_format", "native")

    preds = None
    output_paths = []
    for label_file, vid_file in zip(labels_files, vid_files):
        dataset = pred_cfg.get_dataset(
            label_files=[label_file],
            vid_files=[vid_file],
            mode="test",
            overrides=overrides_dict,
        )
        dataloader = pred_cfg.get_dataloader(dataset, mode="test")
        if isinstance(vid_file, list):
            save_file_name = Path(vid_file[0]).parent.name
        else:
            save_file_name = vid_file

        timestamp = get_timestamp()
        stem = Path(save_file_name).stem

        if isinstance(dataset, CellTrackingDataset):
            preds, all_frames = track_ctc(model, trainer, dataloader)
            if output_format in ("native", "both"):
                outpath = os.path.join(
                    outdir,
                    f"{stem}.dreem_inference.{timestamp}.tif",
                )
                tifffile.imwrite(outpath, preds.astype(np.uint16))
                outpath = os.path.abspath(outpath)
                output_paths.append(outpath)
                print(f"Saved: {outpath}")
        else:
            preds, all_frames = track_sleap(
                model, trainer, dataloader, outdir, overrides_dict
            )
            if output_format in ("native", "both"):
                outpath = os.path.join(
                    outdir,
                    f"{stem}.dreem_inference.{timestamp}.slp",
                )
                preds.save(outpath)
                outpath = os.path.abspath(outpath)
                output_paths.append(outpath)
                print(f"Saved: {outpath}")

        if output_format in ("csv", "both"):
            csv_path = os.path.join(
                outdir,
                f"{stem}.dreem_inference.{timestamp}.csv",
            )
            export_trajectories(all_frames, save_path=csv_path)
            csv_path = os.path.abspath(csv_path)
            output_paths.append(csv_path)
            print(f"Saved: {csv_path}")

        summary = _summarize_preds(preds)
        print(
            f"  Frames: {summary['num_frames']}, "
            f"Tracks: {summary['num_tracks']}, "
            f"IDs: {summary['track_ids']}"
        )

    return {
        "preds": preds,
        "output_paths": output_paths,
        "summary": summary if output_paths else {},
    }


def run_tracking(
    frames: str | np.ndarray,
    masks: str | np.ndarray,
    checkpoint: str,
    crop_size: int = 25,
    output_dir: str = "./results",
    device: str = "auto",
    output_format: str = "both",
    ctc_paths: dict[str, str] | None = None,
    **tracker_overrides,
) -> dict:
    """Run tracking with minimal configuration.

    A convenience wrapper that handles CTC directory setup, config construction,
    and inference in a single call. Accepts flexible input formats for frames
    and masks (directories, TIFF stacks, video files, numpy arrays).

    Args:
        frames: Raw video frames. Can be a path to a directory of TIFFs,
            a TIFF stack, a video file (.mp4/.avi/.mov), a numpy array (T,H,W),
            or a sleap_io.Video object.
        masks: Segmentation masks in the same formats as frames.
        checkpoint: Path to model checkpoint, shortname (e.g., "microscopy"),
            or HuggingFace repo ID.
        crop_size: Bounding box crop size in pixels.
        output_dir: Where to save results and intermediate files.
        device: Accelerator ("auto", "gpu", "cpu", "mps").
        output_format: Output format: "native" (format-specific only), "csv"
            (CSV only), or "both" (default for programmatic use).
        ctc_paths: Pre-built CTC directory paths dict (with keys "raw_dir",
            "dataset_dir", and optionally "mask_dir"). If provided,
            ``setup_ctc_dirs()`` is skipped and these paths are used directly.
        **tracker_overrides: Extra tracker config overrides
            (e.g., window_size=16, overlap_thresh=0.05).

    Returns:
        Dict with "preds" (numpy array of tracked masks), "output_paths"
        (list of absolute path strings), and "summary" (dict with num_frames,
        num_tracks, track_ids).
    """
    from omegaconf import OmegaConf

    if ctc_paths is not None:
        paths = ctc_paths
    else:
        from dreem.utils.ctc_helpers import setup_ctc_dirs

        # Set up CTC directory structure in a cache directory
        cache_dir = os.path.join(output_dir, ".dreem_cache")
        paths = setup_ctc_dirs(frames, masks, output_dir=cache_dir)

    # Load default tracking config
    default_cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "defaults", "track.yaml"
    )
    cfg = OmegaConf.load(default_cfg_path)

    # Configure paths and model
    with OmegaConf.read_write(cfg):
        cfg.ckpt_path = checkpoint
        cfg.outdir = output_dir
        cfg.output_format = output_format
        cfg.dataset.test_dataset.dir.path = paths["dataset_dir"]
        cfg.dataset.test_dataset.dir.labels_suffix = ".tif"
        cfg.dataset.test_dataset.dir.vid_suffix = ".tif"
        cfg.dataset.test_dataset.crop_size = crop_size

        # Map device to Lightning accelerator
        if device == "auto":
            cfg.trainer = {"accelerator": "auto", "devices": 1}
        elif device in ("gpu", "cpu", "mps"):
            cfg.trainer = {"accelerator": device, "devices": 1}

        # Apply tracker overrides
        if tracker_overrides:
            for key, value in tracker_overrides.items():
                OmegaConf.update(cfg, f"tracker.{key}", value)

    return run(cfg)
