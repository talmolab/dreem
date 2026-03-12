"""Helpers for creating Cell Tracking Challenge (CTC) directory structures."""

import os
from pathlib import Path

from dreem.utils.run_cellpose_segmentation import _to_frame_array


def setup_ctc_dirs(
    raw_frames,
    masks=None,
    output_dir=".",
    name=None,
):
    """Create CTC-format directory structure from frames and/or masks.

    Converts raw frames and optional segmentation masks into the standard
    Cell Tracking Challenge directory layout expected by CellTrackingDataset:

        {output_dir}/{name}/frame_00000.tif, frame_00001.tif, ...
        {output_dir}/{name}_GT/TRA/frame_00000.tif, frame_00001.tif, ...

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

    # Load raw frames
    frames_array = _to_frame_array(raw_frames)

    # Create raw frames directory
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
        masks_array = _to_frame_array(masks)
        mask_dir = os.path.join(output_dir, f"{name}_GT", "TRA")
        os.makedirs(mask_dir, exist_ok=True)
        for i, mask in enumerate(masks_array):
            tifffile.imwrite(os.path.join(mask_dir, f"frame_{i:05d}.tif"), mask)
        result["mask_dir"] = os.path.abspath(mask_dir)

    return result
