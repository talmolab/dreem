"""Tests for flexible CellPose inputs and CTC directory helpers."""

import os

import numpy as np
import pytest
import tifffile

from dreem.utils.run_cellpose_segmentation import _to_frame_array

# ---------------------------------------------------------------------------
# _to_frame_array tests
# ---------------------------------------------------------------------------

CELL_TRACKING_DIR = os.path.join(
    os.path.dirname(__file__), "data", "microscopy", "cell_tracking", "test_0"
)


class TestToFrameArrayDirectory:
    """Test loading from a directory of TIFFs."""

    def test_loads_from_directory(self):
        """Load frames from test directory."""
        frames = _to_frame_array(CELL_TRACKING_DIR)
        assert isinstance(frames, np.ndarray)
        assert frames.ndim == 3
        # test_0 has 11 frames
        assert frames.shape[0] == 11

    def test_frame_order_matches_sorted_filenames(self):
        """Verify frame order matches sorted filename order."""
        frames = _to_frame_array(CELL_TRACKING_DIR)
        tiff_files = sorted(
            f for f in os.listdir(CELL_TRACKING_DIR) if f.endswith(".tif")
        )
        first_frame = tifffile.imread(os.path.join(CELL_TRACKING_DIR, tiff_files[0]))
        np.testing.assert_array_equal(frames[0], first_frame)


class TestToFrameArrayNumpy:
    """Test numpy array passthrough."""

    def test_3d_passthrough(self):
        """3D array passes through unchanged."""
        arr = np.random.randint(0, 255, (5, 64, 64), dtype=np.uint8)
        result = _to_frame_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_4d_to_grayscale(self):
        """4D array is converted to grayscale by taking first channel."""
        arr = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        result = _to_frame_array(arr)
        assert result.ndim == 3
        assert result.shape == (5, 64, 64)
        np.testing.assert_array_equal(result, arr[..., 0])

    def test_2d_raises(self):
        """2D array raises ValueError."""
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3D array"):
            _to_frame_array(arr)

    def test_5d_raises(self):
        """5D array raises ValueError."""
        arr = np.zeros((2, 3, 64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3D array"):
            _to_frame_array(arr)


class TestToFrameArrayTiffStack:
    """Test loading from a multi-page TIFF stack."""

    def test_loads_tiff_stack(self, tmp_path):
        """Load from a multi-page TIFF file."""
        stack = np.random.randint(0, 255, (5, 32, 32), dtype=np.uint8)
        tiff_path = str(tmp_path / "stack.tif")
        tifffile.imwrite(tiff_path, stack)

        result = _to_frame_array(tiff_path)
        assert result.ndim == 3
        assert result.shape[0] == 5
        np.testing.assert_array_equal(result, stack)


class TestToFrameArrayVideo:
    """Test loading from a video file."""

    def test_loads_mp4(self, tmp_path):
        """Load from an MP4 video file."""
        imageio = pytest.importorskip("imageio")
        video_path = str(tmp_path / "test.mp4")
        frames = np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8)
        writer = imageio.get_writer(video_path, fps=5)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        result = _to_frame_array(video_path)
        assert result.ndim == 3
        assert result.shape[0] == 10
        assert result.shape[1] == 32
        assert result.shape[2] == 32


class TestToFrameArrayErrors:
    """Test error handling."""

    def test_nonexistent_path(self):
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            _to_frame_array("/nonexistent/path/video.tif")

    def test_unsupported_type(self):
        """Unsupported input type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            _to_frame_array(42)


# ---------------------------------------------------------------------------
# run_cellpose_segmentation tests
# ---------------------------------------------------------------------------


class TestSegmentationNoOutput:
    """Test segmentation without writing output files."""

    def test_returns_array_no_write(self, tmp_path):
        """Return masks array without writing to disk."""
        pytest.importorskip("cellpose")
        from dreem.utils.run_cellpose_segmentation import run_cellpose_segmentation

        arr = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        result = run_cellpose_segmentation(
            arr, output_path=None, diameter=15, gpu=False
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 64, 64)


class TestSegmentationWithOutput:
    """Test segmentation with file output (backward compat)."""

    def test_writes_files(self, tmp_path):
        """Write mask files to disk and verify filenames."""
        pytest.importorskip("cellpose")
        from dreem.utils.run_cellpose_segmentation import run_cellpose_segmentation

        arr = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        output = str(tmp_path / "masks")
        result = run_cellpose_segmentation(
            arr, output_path=output, diameter=15, gpu=False
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 64, 64)
        # Verify files were written
        written_files = sorted(os.listdir(output))
        assert len(written_files) == 3
        assert written_files[0] == "frame_00000.tif"


# ---------------------------------------------------------------------------
# setup_ctc_dirs tests
# ---------------------------------------------------------------------------


class TestSetupCtcDirsArrays:
    """Test CTC directory setup from numpy arrays."""

    def test_creates_structure(self, tmp_path):
        """Create full CTC directory structure from arrays."""
        from dreem.utils.ctc_helpers import setup_ctc_dirs

        frames = np.random.randint(0, 255, (5, 32, 32), dtype=np.uint8)
        masks = np.random.randint(0, 10, (5, 32, 32), dtype=np.uint16)

        result = setup_ctc_dirs(
            frames, masks, output_dir=str(tmp_path), name="test_seq"
        )

        # Verify directory structure
        assert os.path.isdir(os.path.join(tmp_path, "test_seq"))
        assert os.path.isdir(os.path.join(tmp_path, "test_seq_GT", "TRA"))

        # Verify files
        raw_files = sorted(os.listdir(os.path.join(tmp_path, "test_seq")))
        mask_files = sorted(os.listdir(os.path.join(tmp_path, "test_seq_GT", "TRA")))
        assert len(raw_files) == 5
        assert len(mask_files) == 5
        assert raw_files[0] == "frame_00000.tif"
        assert mask_files[0] == "frame_00000.tif"

        # Verify return dict
        assert "raw_dir" in result
        assert "mask_dir" in result
        assert "dataset_dir" in result

    def test_frames_only(self, tmp_path):
        """Create CTC structure with raw frames only (no masks)."""
        from dreem.utils.ctc_helpers import setup_ctc_dirs

        frames = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        result = setup_ctc_dirs(frames, output_dir=str(tmp_path), name="raw_only")

        assert os.path.isdir(os.path.join(tmp_path, "raw_only"))
        assert not os.path.exists(os.path.join(tmp_path, "raw_only_GT"))
        assert "mask_dir" not in result

    def test_auto_name_from_array(self, tmp_path):
        """Default name 'dataset' is used for numpy array input."""
        from dreem.utils.ctc_helpers import setup_ctc_dirs

        frames = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        setup_ctc_dirs(frames, output_dir=str(tmp_path))

        # Default name for arrays is "dataset"
        assert os.path.isdir(os.path.join(tmp_path, "dataset"))


class TestSetupCtcDirsFromFile:
    """Test CTC directory setup from file paths."""

    def test_from_tiff_stack(self, tmp_path):
        """Extract frames from a TIFF stack into CTC structure."""
        from dreem.utils.ctc_helpers import setup_ctc_dirs

        # Create a TIFF stack
        stack = np.random.randint(0, 255, (4, 32, 32), dtype=np.uint8)
        tiff_path = str(tmp_path / "source" / "my_video.tif")
        os.makedirs(os.path.dirname(tiff_path), exist_ok=True)
        tifffile.imwrite(tiff_path, stack)

        out_dir = str(tmp_path / "output")
        setup_ctc_dirs(tiff_path, output_dir=out_dir)

        # Name should be auto-derived from filename: "my_video"
        assert os.path.isdir(os.path.join(out_dir, "my_video"))
        raw_files = sorted(os.listdir(os.path.join(out_dir, "my_video")))
        assert len(raw_files) == 4

    def test_from_directory(self, tmp_path):
        """Extract frames from a directory of TIFFs into CTC structure."""
        from dreem.utils.ctc_helpers import setup_ctc_dirs

        # Create a directory of TIFFs
        src_dir = str(tmp_path / "source" / "my_frames")
        os.makedirs(src_dir)
        for i in range(3):
            tifffile.imwrite(
                os.path.join(src_dir, f"img_{i:03d}.tif"),
                np.random.randint(0, 255, (32, 32), dtype=np.uint8),
            )

        out_dir = str(tmp_path / "output")
        setup_ctc_dirs(src_dir, output_dir=out_dir)

        # Name should be auto-derived from directory name: "my_frames"
        assert os.path.isdir(os.path.join(out_dir, "my_frames"))


class TestSetupCtcDirsCompat:
    """Test that setup_ctc_dirs output is compatible with CellTrackingDataset."""

    def test_paths_compatible_with_get_ctc_paths(self, tmp_path):
        """Verify the directory structure matches what get_ctc_paths expects."""
        from dreem.utils.ctc_helpers import setup_ctc_dirs

        frames = np.random.randint(0, 255, (5, 32, 32), dtype=np.uint8)
        masks = np.random.randint(0, 10, (5, 32, 32), dtype=np.uint16)

        result = setup_ctc_dirs(frames, masks, output_dir=str(tmp_path), name="test_0")

        # Verify the structure matches CTC convention:
        # {dataset_dir}/test_0/*.tif (raw)
        # {dataset_dir}/test_0_GT/TRA/*.tif (masks)
        dataset_dir = result["dataset_dir"]

        # Check that _GT directory exists
        gt_dir = os.path.join(dataset_dir, "test_0_GT")
        assert os.path.isdir(gt_dir)

        # Check TRA subdirectory
        tra_dir = os.path.join(gt_dir, "TRA")
        assert os.path.isdir(tra_dir)

        # Check raw directory
        raw_dir = os.path.join(dataset_dir, "test_0")
        assert os.path.isdir(raw_dir)

        # Verify files match between raw and mask directories
        import glob

        raw_tifs = sorted(glob.glob(os.path.join(raw_dir, "*.tif")))
        mask_tifs = sorted(glob.glob(os.path.join(tra_dir, "*.tif")))
        assert len(raw_tifs) == len(mask_tifs) == 5
