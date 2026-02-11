
r"""Helper script to run CellPose segmentation using uv for dependency management.

This allows running CellPose without global installation.

Usage:
    uv run run_cellpose_segmentation.py \\
        --data_path ./data/dynamicnuclearnet/test_1 \\
        --output_path ./data/dynamicnuclearnet/test_1_GT/TRA \\
        --diameter 25

Or use as a module:
    from run_cellpose_segmentation import run_cellpose_segmentation
    run_cellpose_segmentation(data_path, output_path, diameter=25, gpu=True)
"""

import os
import numpy as np

def run_cellpose_segmentation(
    data_path,
    output_path,
    diameter=25,
    gpu=True,
    cellprob_threshold=0.0,
):
    """Run CellPose segmentation on a directory of tiff files.

    Parameters
    ----------
    data_path : str
        Path to directory containing input tiff files
    output_path : str
        Path to directory where segmentation masks will be saved
    diameter : int, default=25
        Approximate diameter (in pixels) of instances to segment
    gpu : bool, default=True
        Use GPU if available
    cellprob_threshold : float, default=0.0
        Cell probability threshold

    Returns:
    --------
    all_masks : numpy.ndarray
        Array of segmentation masks
    """
    import tifffile
    from cellpose import models
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load tiff files
    tiff_files = [
        f for f in os.listdir(data_path) if f.endswith(".tif") or f.endswith(".tiff")
    ]
    tiff_files.sort()  # Ensure consistent ordering

    if not tiff_files:
        raise ValueError(f"No tiff files found in {data_path}")

    print(f"Loading {len(tiff_files)} tiff files from {data_path}...")
    stack = np.stack([tifffile.imread(os.path.join(data_path, f)) for f in tiff_files])
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

    # Save segmentation masks
    print(f"Saving masks to {output_path}...")
    for i, (mask, filename) in enumerate(zip(all_masks, tiff_files)):
        new_tiff_path = os.path.join(
            output_path, f"{os.path.splitext(filename)[0]}.tif"
        )
        tifffile.imwrite(new_tiff_path, mask)
        print(f"Saved frame {i + 1} to {new_tiff_path}")

    print("Segmentation complete!")
    return all_masks


# def main():
#     """Command-line interface for CellPose segmentation."""
#     parser = argparse.ArgumentParser(
#         description="Run CellPose segmentation on a directory of tiff files"
#     )
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         required=True,
#         help="Path to directory containing input tiff files",
#     )
#     parser.add_argument(
#         "--output_path",
#         type=str,
#         required=True,
#         help="Path to directory where segmentation masks will be saved",
#     )
#     parser.add_argument(
#         "--diameter",
#         type=int,
#         default=25,
#         help="Approximate diameter (in pixels) of instances to segment (default: 25)",
#     )
#     parser.add_argument(
#         "--gpu",
#         action="store_true",
#         default=True,
#         help="Use GPU if available (default: True)",
#     )
#     parser.add_argument(
#         "--no-gpu",
#         dest="gpu",
#         action="store_false",
#         help="Disable GPU usage",
#     )
#     parser.add_argument(
#         "--cellprob_threshold",
#         type=float,
#         default=0.0,
#         help="Cell probability threshold (default: 0.0)",
#     )

#     args = parser.parse_args()

#     run_cellpose_segmentation(
#         data_path=args.data_path,
#         output_path=args.output_path,
#         diameter=args.diameter,
#         gpu=args.gpu,
#         cellprob_threshold=args.cellprob_threshold,
#     )


# if __name__ == "__main__":
#     main()
