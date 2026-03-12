"""Helpers for creating Cell Tracking Challenge (CTC) directory structures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from dreem.utils.run_cellpose_segmentation import load_frames


def setup_ctc_dirs(
    raw_frames: str | np.ndarray,
    masks: str | np.ndarray | None = None,
    output_dir: str = ".",
    name: str | None = None,
) -> dict[str, str]:
    """Create CTC-format directory structure from frames and/or masks.

    Converts raw frames and optional segmentation masks into the standard
    Cell Tracking Challenge directory layout expected by CellTrackingDataset:

        {output_dir}/{name}/frame_00000.tif, frame_00001.tif, ...
        {output_dir}/{name}_GT/TRA/frame_00000.tif, frame_00001.tif, ...

    When ``raw_frames`` or ``masks`` is a path to an existing directory of TIFFs,
    that directory is used directly instead of copying files.

    Args:
        raw_frames: Raw video frames. Can be a path to a directory of TIFFs,
            a TIFF stack, a video file, a numpy array (T,H,W), or a
            sleap_io.Video object.
        masks: Segmentation masks in the same formats as raw_frames.
            If None, only the raw frames directory is created.
        output_dir: Parent directory for the CTC structure.
        name: Dataset name used for directory naming. If None, auto-derived
            from the source path (e.g., "video.mp4" -> "video") or defaults
            to "dataset".

    Returns:
        Dict with keys "raw_dir", "dataset_dir", and optionally "mask_dir"
        containing absolute path strings.

    Raises:
        ValueError: If the number of raw frames and masks don't match.
    """
    import tifffile

    # Auto-derive name from source path if not provided
    if name is None:
        if isinstance(raw_frames, str):
            name = Path(raw_frames).stem
            # If it's a directory, use the directory name
            if os.path.isdir(raw_frames):
                name = Path(raw_frames).name
        else:
            name = "dataset"

    # Load raw frames (or use existing directory directly)
    if isinstance(raw_frames, str) and os.path.isdir(raw_frames):
        # Directory of TIFFs already exists — use it directly
        raw_dir = raw_frames
        frames_array = None
    else:
        frames_array = load_frames(raw_frames)
        raw_dir = os.path.join(output_dir, name)
        os.makedirs(raw_dir, exist_ok=True)
        for i, frame in enumerate(frames_array):
            tifffile.imwrite(os.path.join(raw_dir, f"frame_{i:05d}.tif"), frame)

    result = {
        "raw_dir": os.path.abspath(raw_dir),
        "dataset_dir": os.path.abspath(output_dir),
    }

    # Create mask directory if masks provided
    if masks is not None:
        if isinstance(masks, str) and os.path.isdir(masks):
            # Directory of mask TIFFs already exists — use it directly
            mask_dir = masks
            masks_array = None
        else:
            masks_array = load_frames(masks)
            mask_dir = os.path.join(output_dir, f"{name}_GT", "TRA")
            os.makedirs(mask_dir, exist_ok=True)
            for i, mask in enumerate(masks_array):
                tifffile.imwrite(os.path.join(mask_dir, f"frame_{i:05d}.tif"), mask)

        # Validate frame/mask count agreement
        if frames_array is None:
            n_raw = len(
                [f for f in os.listdir(raw_frames) if f.endswith((".tif", ".tiff"))]
            )
        else:
            n_raw = frames_array.shape[0]

        if masks_array is None:
            n_masks = len(
                [f for f in os.listdir(masks) if f.endswith((".tif", ".tiff"))]
            )
        else:
            n_masks = masks_array.shape[0]

        if n_raw != n_masks:
            raise ValueError(
                f"Frame count mismatch: raw_frames has {n_raw} frames "
                f"but masks has {n_masks} frames. "
                f"raw_frames source: {raw_frames!r}, masks source: {masks!r}"
            )

        result["mask_dir"] = os.path.abspath(mask_dir)

    return result
