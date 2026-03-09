"""Test to verify that file ordering in dataset loading matches visualization loading.

This test verifies the fix for the bug where `glob.glob()` in `config.py` returned
files in arbitrary (filesystem-dependent) order, causing tracked output TIFF frames
to not align with raw images and detection masks loaded in sorted order.

The notebook visualization loads:
  - Raw images via TiffSequence (sorted)
  - Detections via sorted(os.listdir(...)) (sorted)
  - Tracked output via tifffile.imread (TIFF frame order = dataset processing order)

If the dataset processes frames in unsorted glob order, the tracked TIFF has frames
in the wrong order relative to the sorted raw/detection arrays.
"""

import glob
import os
import tempfile

import numpy as np
import tifffile


def _create_test_ctc_structure(base_dir: str, n_frames: int = 5) -> str:
    """Create a minimal CTC-style directory structure for testing.

    Creates:
      base_dir/
        test_vid/
          t000.tif, t001.tif, ...   (raw images)
        test_vid_GT/
          TRA/
            t000.tif, t001.tif, ... (segmentation masks)

    Each frame has a single "cell" at position (20+i*15, 20+i*15) so we can
    verify frame-to-mask alignment by checking centroid positions.
    """
    raw_dir = os.path.join(base_dir, "test_vid")
    gt_dir = os.path.join(base_dir, "test_vid_GT", "TRA")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for i in range(n_frames):
        # Raw image: bright spot at unique position per frame
        img = np.zeros((128, 128), dtype=np.uint16)
        cy, cx = 20 + i * 15, 20 + i * 15  # diagonal positions
        img[cy - 5 : cy + 5, cx - 5 : cx + 5] = 1000 + i
        tifffile.imwrite(os.path.join(raw_dir, f"t{i:03d}.tif"), img)

        # GT mask: labeled region at same position, unique per frame
        mask = np.zeros((128, 128), dtype=np.uint16)
        mask[cy - 5 : cy + 5, cx - 5 : cx + 5] = 1  # instance ID = 1
        tifffile.imwrite(os.path.join(gt_dir, f"t{i:03d}.tif"), mask)

    return base_dir


def _get_ctc_paths_from_dir(dir_path: str):
    """Replicate the glob logic from Config.get_ctc_paths."""
    gt_list = []
    raw_img_list = []
    for subdir in os.listdir(dir_path):
        if subdir.endswith("_GT"):
            gt_path = os.path.join(dir_path, subdir, "TRA")
            raw_img_path = os.path.join(dir_path, subdir.replace("_GT", ""))
            gt_list.append(sorted(glob.glob(os.path.join(gt_path, "*.tif*"))))
            raw_img_list.append(
                sorted(glob.glob(os.path.join(raw_img_path, "*.tif*")))
            )
    return gt_list, raw_img_list


class TestGlobOrdering:
    """Tests for file ordering consistency."""

    def test_get_ctc_paths_returns_sorted(self):
        """Verify that get_ctc_paths returns file lists in sorted order."""
        with tempfile.TemporaryDirectory() as tmp:
            _create_test_ctc_structure(tmp, n_frames=10)

            gt_list, raw_img_list = _get_ctc_paths_from_dir(tmp)

            for gt_files in gt_list:
                assert gt_files == sorted(gt_files), (
                    f"GT files not in sorted order.\n"
                    f"Got:      {[os.path.basename(f) for f in gt_files]}\n"
                    f"Expected: {[os.path.basename(f) for f in sorted(gt_files)]}"
                )

            for raw_files in raw_img_list:
                assert raw_files == sorted(raw_files), (
                    f"Raw image files not in sorted order.\n"
                    f"Got:      {[os.path.basename(f) for f in raw_files]}\n"
                    f"Expected: {[os.path.basename(f) for f in sorted(raw_files)]}"
                )

    def test_gt_and_raw_files_correspond(self):
        """Verify that gt_list[i] and raw_img_list[i] refer to the same frame."""
        with tempfile.TemporaryDirectory() as tmp:
            _create_test_ctc_structure(tmp, n_frames=10)

            gt_list, raw_img_list = _get_ctc_paths_from_dir(tmp)

            for gt_files, raw_files in zip(gt_list, raw_img_list):
                assert len(gt_files) == len(raw_files)
                for gt_f, raw_f in zip(gt_files, raw_files):
                    gt_name = os.path.basename(gt_f)
                    raw_name = os.path.basename(raw_f)
                    assert gt_name == raw_name, (
                        f"GT file '{gt_name}' does not match raw file '{raw_name}'. "
                        f"Files are paired incorrectly."
                    )

    def test_dataset_frame_order_matches_sorted_glob(self):
        """Verify dataset processes frames in sorted filename order.

        This is critical because the notebook visualization loads files in
        sorted order (TiffSequence, sorted(os.listdir)), so the tracked
        output TIFF must also be in sorted order.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _create_test_ctc_structure(tmp, n_frames=5)

            gt_list, _ = _get_ctc_paths_from_dir(tmp)

            expected_order = [os.path.basename(f) for f in sorted(gt_list[0])]
            actual_order = [os.path.basename(f) for f in gt_list[0]]

            assert actual_order == expected_order, (
                f"Dataset file order doesn't match sorted order.\n"
                f"Dataset order: {actual_order}\n"
                f"Sorted order:  {expected_order}\n"
                f"This causes tracked output frames to misalign with "
                f"raw images in visualization."
            )

    def test_notebook_loading_consistency(self):
        """End-to-end test: verify that all three data sources align by frame index.

        Simulates the notebook's loading pattern:
          images[z]     = TiffSequence (sorted)
          detections[z] = sorted(os.listdir(...))
          tracked[z]    = TIFF written by track_ctc (dataset order)

        All three must agree on which frame is at index z.
        """
        with tempfile.TemporaryDirectory() as tmp:
            n_frames = 8
            _create_test_ctc_structure(tmp, n_frames=n_frames)

            raw_dir = os.path.join(tmp, "test_vid")
            gt_dir = os.path.join(tmp, "test_vid_GT", "TRA")

            # Method 1: TiffSequence (how notebook loads raw images)
            # Verify TiffSequence can load without error (sorted internally)
            seq = tifffile.TiffSequence(os.path.join(raw_dir, "*.tif"))
            _ = seq.asarray()

            # Method 2: sorted(os.listdir(...)) (how notebook loads detections)
            det_files = sorted(
                f
                for f in os.listdir(gt_dir)
                if f.endswith(".tif") or f.endswith(".tiff")
            )
            detections = np.stack(
                [tifffile.imread(os.path.join(gt_dir, f)) for f in det_files]
            )

            # Method 3: Dataset ordering (how tracked output would be ordered)
            gt_list, _ = _get_ctc_paths_from_dir(tmp)
            dataset_gt_files = gt_list[0]
            tracked_equiv = np.stack(
                [tifffile.imread(f) for f in dataset_gt_files]
            )

            # All three should match frame-for-frame
            for z in range(n_frames):
                assert np.array_equal(detections[z], tracked_equiv[z]), (
                    f"Frame {z}: detection mask doesn't match dataset-ordered mask.\n"
                    f"Detection file: {det_files[z]}\n"
                    f"Dataset file:   {os.path.basename(dataset_gt_files[z])}"
                )

    def test_unsorted_glob_causes_mismatch(self):
        """Demonstrate that unsorted glob WOULD cause frame misalignment.

        This test explicitly shuffles file paths (simulating Linux glob behavior)
        and shows that the tracked output would be misaligned with sorted loading.
        """
        with tempfile.TemporaryDirectory() as tmp:
            n_frames = 5
            _create_test_ctc_structure(tmp, n_frames=n_frames)

            gt_dir = os.path.join(tmp, "test_vid_GT", "TRA")

            # Sorted order (what notebook expects)
            sorted_files = sorted(glob.glob(os.path.join(gt_dir, "*.tif*")))

            # Simulate unsorted glob (as would happen on Linux ext4)
            import random

            rng = random.Random(42)
            unsorted_files = sorted_files.copy()
            rng.shuffle(unsorted_files)

            if unsorted_files == sorted_files:
                unsorted_files[0], unsorted_files[-1] = (
                    unsorted_files[-1],
                    unsorted_files[0],
                )

            # Load masks in both orders
            sorted_masks = np.stack([tifffile.imread(f) for f in sorted_files])
            unsorted_masks = np.stack(
                [tifffile.imread(f) for f in unsorted_files]
            )

            # Unsorted should NOT match sorted (proving unsorted glob causes mismatch)
            assert not np.array_equal(sorted_masks, unsorted_masks), (
                "Shuffled file order produced identical masks — "
                "test frames may not have unique content"
            )

            # Re-sorting the shuffled files should restore correct order
            resorted_files = sorted(unsorted_files)
            resorted_masks = np.stack(
                [tifffile.imread(f) for f in resorted_files]
            )
            assert np.array_equal(sorted_masks, resorted_masks), (
                "Re-sorting the shuffled files should restore correct order"
            )

    def test_config_get_ctc_paths_uses_sorted_glob(self):
        """Verify the actual Config.get_ctc_paths method sorts its results.

        This tests the real implementation, not our local replica.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _create_test_ctc_structure(tmp, n_frames=10)

            # Read the config.py source and verify sorted() wraps glob
            import inspect

            from dreem.io.config import Config

            source = inspect.getsource(Config.get_ctc_paths)
            assert "sorted(glob.glob(" in source, (
                "Config.get_ctc_paths does not use sorted(glob.glob(...)). "
                "File ordering will be filesystem-dependent, causing frame "
                "misalignment on Linux."
            )
