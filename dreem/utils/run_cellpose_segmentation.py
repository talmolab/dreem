r"""Utilities for running CellPose segmentation and loading image data.

Provides:
    load_frames(): Convert directories, TIFF stacks, videos, or arrays to (T,H,W).
    run_cellpose_segmentation(): Run CellPose on any supported input format.

Example:
    from dreem.utils import load_frames, run_cellpose_segmentation
    masks = run_cellpose_segmentation("./data/my_tiffs", diameter=25, gpu=True)
"""

from __future__ import annotations

import os

import numpy as np


def load_frames(
    data: str | np.ndarray,
    grayscale: str | bool = "first_channel",
) -> np.ndarray:
    """Convert any supported input to a (T, H, W) numpy array.

    Supports: directory of TIFFs, TIFF stack, video file (.mp4/.avi/.mov),
    numpy array (T,H,W) or (T,H,W,C), or sleap_io.Video object.

    Args:
        data: Input data as a file path (str), numpy array, or sleap_io.Video.
        grayscale: How to handle color channels.
            - ``"first_channel"`` (default): For 4D arrays, take ``data[..., 0]``.
              For video/TIFF files loaded via sleap_io, load as grayscale.
            - ``"luminance"``: For 4D arrays, compute weighted sum
              ``0.299*R + 0.587*G + 0.114*B``. For video/TIFF files loaded via
              sleap_io, load as grayscale (sleap-io handles conversion).
            - ``False``: Return as-is with no channel conversion. 4D arrays stay
              4D, and videos are loaded in color.

    Returns:
        A numpy array with shape (T, H, W) when grayscale is enabled, or
        (T, H, W, C) when ``grayscale=False`` and the input has color channels.

    Raises:
        ValueError: If the input array has unsupported dimensions.
        FileNotFoundError: If the input path does not exist.
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 4 and grayscale != False:  # noqa: E712
            if grayscale == "luminance" and data.shape[-1] >= 3:
                data = (
                    0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
                ).astype(data.dtype)
            else:
                # "first_channel" or fallback
                data = data[..., 0]
        if grayscale != False and data.ndim != 3:  # noqa: E712
            raise ValueError(f"Expected 3D array (T, H, W), got {data.ndim}D")
        if grayscale == False and data.ndim not in (3, 4):  # noqa: E712
            raise ValueError(
                f"Expected 3D or 4D array (T, H, W[, C]), got {data.ndim}D"
            )
        return data

    import sleap_io as sio

    load_grayscale = grayscale != False  # noqa: E712

    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"Path does not exist: {data}")
        video = sio.load_video(data, grayscale=load_grayscale)
    elif isinstance(data, sio.Video):
        video = data
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected str, np.ndarray, or sleap_io.Video."
        )

    frames = video[:]  # (T, H, W, 1) for grayscale or (T, H, W) or (T, H, W, C)
    if load_grayscale and frames.ndim == 4:
        frames = np.squeeze(frames, axis=-1)  # (T, H, W)
    return frames


def run_cellpose_segmentation(
    data: str | np.ndarray,
    output_path: str | None = None,
    diameter: int = 25,
    gpu: bool = True,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
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
    stack = load_frames(data)
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
