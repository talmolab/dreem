r"""Helper script to run CellPose segmentation using uv for dependency management.

This allows running CellPose without global installation.

Usage:
    uv run run_cellpose_segmentation.py \\
        --data_path ./data/dynamicnuclearnet/test_1 \\
        --output_path ./data/dynamicnuclearnet/test_1_GT/TRA \\
        --diameter 25

Or use as a module:
    from dreem.utils import run_cellpose_segmentation
    masks = run_cellpose_segmentation(data_path, diameter=25, gpu=True)
"""

import os

import numpy as np


def _to_frame_array(data):
    """Convert any supported input to a (T, H, W) grayscale numpy array.

    Supports: directory of TIFFs, TIFF stack, video file (.mp4/.avi/.mov),
    numpy array (T,H,W) or (T,H,W,C), or sleap_io.Video object.

    Args:
        data: Input data as a file path (str), numpy array, or sleap_io.Video.

    Returns:
        A numpy array with shape (T, H, W).

    Raises:
        ValueError: If the input array has unsupported dimensions.
        FileNotFoundError: If the input path does not exist.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 4:  # (T, H, W, C) -> grayscale
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array (T, H, W), got {data.ndim}D")
        return data

    import sleap_io as sio

    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"Path does not exist: {data}")
        video = sio.load_video(data, grayscale=True)
    elif isinstance(data, sio.Video):
        video = data
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected str, np.ndarray, or sleap_io.Video."
        )

    frames = video[:]  # (T, H, W, 1) for grayscale or (T, H, W)
    if frames.ndim == 4:
        frames = np.squeeze(frames, axis=-1)  # (T, H, W)
    return frames


def run_cellpose_segmentation(
    data,
    output_path=None,
    diameter=25,
    gpu=True,
    cellprob_threshold=0.0,
):
    """Run CellPose segmentation on image data.

    Supports directories of TIFFs, TIFF stacks, video files, numpy arrays,
    and sleap_io.Video objects as input.

    Args:
        data: Input data. Can be a path to a directory of TIFFs, a TIFF stack,
            a video file (.mp4/.avi/.mov), a numpy array (T,H,W) or (T,H,W,C),
            or a sleap_io.Video object.
        output_path: Path to directory where segmentation masks will be saved.
            If None, masks are returned without writing to disk.
        diameter: Approximate diameter (in pixels) of instances to segment.
        gpu: Use GPU if available.
        cellprob_threshold: Cell probability threshold.

    Returns:
        Numpy array of segmentation masks with shape (T, H, W).
    """
    from cellpose import models

    # Load frames from any supported input format
    stack = _to_frame_array(data)
    frames, Y, X = stack.shape
    print(f"Loaded stack: {frames} frames, {Y}x{X} pixels")

    # Initialize CellPose model
    print(f"Initializing CellPose model (GPU: {gpu})...")
    model = models.CellposeModel(gpu=gpu)

    # Run segmentation on each frame
    print(f"Running segmentation with diameter={diameter}...")
    channels = [0, 0]  # Grayscale channels
    all_masks = np.zeros_like(stack)

    for i, img in enumerate(stack):
        print(f"Processing frame {i + 1}/{frames}...")
        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            channels=channels,
            z_axis=None,
        )
        all_masks[i] = masks

    # Save segmentation masks if output_path is provided
    if output_path is not None:
        import tifffile

        os.makedirs(output_path, exist_ok=True)
        print(f"Saving masks to {output_path}...")
        for i, mask in enumerate(all_masks):
            tiff_path = os.path.join(output_path, f"frame_{i:05d}.tif")
            tifffile.imwrite(tiff_path, mask)
            print(f"Saved frame {i + 1} to {tiff_path}")

    print("Segmentation complete!")
    return all_masks
