"""Tests for tracking visualization functions."""

import tempfile

import imageio
import numpy as np
import pytest
import sleap_io as sio
import tifffile

# Skip all tests if skia-python is not installed
skia = pytest.importorskip("skia", reason="skia-python not installed")

from dreem.io.visualize import (  # noqa: E402
    _blend_mask_overlay,
    _extract_centroids_from_labels,
    extract_centroids_from_masks,
    masks_to_sleap_labels,
    render_ctc_video,
    render_slp_video,
)


def _make_synthetic_masks(
    n_frames: int = 5, height: int = 64, width: int = 64, n_tracks: int = 3
) -> np.ndarray:
    """Create synthetic mask stack with known track positions.

    Each track is a 10x10 square that moves rightward over time.
    """
    masks = np.zeros((n_frames, height, width), dtype=np.uint16)
    for t in range(n_frames):
        for tid in range(1, n_tracks + 1):
            # Each track starts at y = tid * 15, x = 5 + t * 3
            y0 = tid * 15
            x0 = 5 + t * 3
            y1 = min(y0 + 10, height)
            x1 = min(x0 + 10, width)
            masks[t, y0:y1, x0:x1] = tid
    return masks


def _make_temp_video(n_frames: int, height: int, width: int) -> sio.Video:
    """Create a temporary mp4 video file and return a sleap-io Video."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = imageio.get_writer(tmp.name, fps=10)
    for _ in range(n_frames):
        writer.append_data(np.zeros((height, width, 3), dtype=np.uint8))
    writer.close()
    return sio.Video(tmp.name)


def _make_synthetic_labels(
    n_frames: int = 5,
    n_tracks: int = 3,
    n_nodes: int = 2,
    height: int = 64,
    width: int = 64,
) -> sio.Labels:
    """Create synthetic Labels with known keypoint positions.

    Each track has keypoints that move rightward over time.
    """
    skeleton = sio.Skeleton(nodes=[f"node_{i}" for i in range(n_nodes)])
    if n_nodes >= 2:
        skeleton.add_edge("node_0", "node_1")
    tracks = [sio.Track(name=str(i)) for i in range(n_tracks)]
    video = _make_temp_video(n_frames, height, width)
    labeled_frames = []

    for t in range(n_frames):
        instances = []
        for tid in range(n_tracks):
            # Each track starts at different y, moves rightward
            base_x = 10 + t * 3
            base_y = (tid + 1) * 15
            pts = np.array(
                [[base_x + i * 5, base_y + i * 2] for i in range(n_nodes)],
                dtype=np.float64,
            )
            inst = sio.Instance.from_numpy(pts, skeleton=skeleton, track=tracks[tid])
            instances.append(inst)
        lf = sio.LabeledFrame(video=video, frame_idx=t, instances=instances)
        labeled_frames.append(lf)

    return sio.Labels(
        labeled_frames=labeled_frames,
        skeletons=[skeleton],
        tracks=tracks,
    )


class TestExtractCentroidsFromMasks:
    """Tests for extract_centroids_from_masks."""

    def test_basic(self):
        """Test basic centroid extraction with multiple tracks."""
        masks = _make_synthetic_masks(n_frames=3, n_tracks=2)
        centroids = extract_centroids_from_masks(masks)

        assert len(centroids) == 3  # 3 frames
        assert 1 in centroids[0]
        assert 2 in centroids[0]

        # Check centroid values are reasonable (x, y format)
        cx, cy = centroids[0][1]
        assert 5 < cx < 15  # x should be around 10
        assert 15 < cy < 25  # y should be around 20

    def test_empty_frame(self):
        """Test centroid extraction with empty frames."""
        masks = np.zeros((3, 64, 64), dtype=np.uint16)
        centroids = extract_centroids_from_masks(masks)
        assert len(centroids) == 3
        for t in range(3):
            assert centroids[t] == {}

    def test_single_pixel(self):
        """Test centroid extraction with a single pixel mask."""
        masks = np.zeros((1, 32, 32), dtype=np.uint16)
        masks[0, 10, 20] = 5
        centroids = extract_centroids_from_masks(masks)
        cx, cy = centroids[0][5]
        assert cx == pytest.approx(20.0)
        assert cy == pytest.approx(10.0)

    def test_returns_xy_not_rowcol(self):
        """Verify centroids are (x, y) not (row, col)."""
        masks = np.zeros((1, 100, 200), dtype=np.uint16)
        # Place a block at top-right: rows 0-9, cols 190-199
        masks[0, 0:10, 190:200] = 1
        centroids = extract_centroids_from_masks(masks)
        cx, cy = centroids[0][1]
        assert cx > 100  # x should be ~194.5 (right side)
        assert cy < 10  # y should be ~4.5 (top)


class TestExtractCentroidsFromLabels:
    """Tests for _extract_centroids_from_labels."""

    def test_basic(self):
        """Test basic centroid extraction from Labels."""
        labels = _make_synthetic_labels(n_frames=3, n_tracks=2, n_nodes=2)
        centroids = _extract_centroids_from_labels(labels)

        assert len(centroids) == 3  # 3 frames
        assert 0 in centroids[0]  # track index 0
        assert 1 in centroids[0]  # track index 1

    def test_centroid_values(self):
        """Test that centroids are mean of keypoints."""
        labels = _make_synthetic_labels(n_frames=1, n_tracks=1, n_nodes=2)
        centroids = _extract_centroids_from_labels(labels)

        # First track, frame 0: nodes at (10, 15) and (15, 17)
        cx, cy = centroids[0][0]
        assert cx == pytest.approx(12.5)  # mean of 10 and 15
        assert cy == pytest.approx(16.0)  # mean of 15 and 17

    def test_empty_labels(self):
        """Test centroid extraction from empty Labels."""
        labels = sio.Labels()
        centroids = _extract_centroids_from_labels(labels)
        assert centroids == {}

    def test_instances_without_tracks_skipped(self):
        """Test that untracked instances are excluded."""
        skeleton = sio.Skeleton(nodes=["centroid"])
        video = _make_temp_video(1, 32, 32)

        # Create one tracked and one untracked instance
        track = sio.Track(name="0")
        tracked_inst = sio.Instance.from_numpy(
            np.array([[10.0, 20.0]]), skeleton=skeleton, track=track
        )
        untracked_inst = sio.Instance.from_numpy(
            np.array([[30.0, 40.0]]), skeleton=skeleton
        )
        lf = sio.LabeledFrame(
            video=video, frame_idx=0, instances=[tracked_inst, untracked_inst]
        )
        labels = sio.Labels(labeled_frames=[lf], skeletons=[skeleton], tracks=[track])

        centroids = _extract_centroids_from_labels(labels)
        assert len(centroids[0]) == 1  # only tracked instance
        assert 0 in centroids[0]

    def test_nan_points_excluded(self):
        """Test that NaN keypoints are excluded from centroid calculation."""
        skeleton = sio.Skeleton(nodes=["head", "tail"])
        track = sio.Track(name="0")
        video = _make_temp_video(1, 64, 64)

        pts = np.array([[10.0, 20.0], [np.nan, np.nan]])
        inst = sio.Instance.from_numpy(pts, skeleton=skeleton, track=track)
        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
        labels = sio.Labels(labeled_frames=[lf], skeletons=[skeleton], tracks=[track])

        centroids = _extract_centroids_from_labels(labels)
        cx, cy = centroids[0][0]
        assert cx == pytest.approx(10.0)
        assert cy == pytest.approx(20.0)


class TestMasksToSleapLabels:
    """Tests for masks_to_sleap_labels."""

    def test_basic_structure(self):
        """Test that Labels has correct structure."""
        masks = _make_synthetic_masks(n_frames=3, n_tracks=2)
        labels = masks_to_sleap_labels(masks)

        assert len(labels) == 3  # 3 labeled frames
        assert len(labels.tracks) == 2
        assert len(labels.skeletons) == 1
        assert labels.skeletons[0].node_names == ["centroid"]

    def test_instances_have_tracks(self):
        """Test that all instances have track assignments."""
        masks = _make_synthetic_masks(n_frames=2, n_tracks=2)
        labels = masks_to_sleap_labels(masks)

        for lf in labels:
            for inst in lf.instances:
                assert inst.track is not None

    def test_masks_present(self):
        """Test that segmentation masks are created."""
        masks = _make_synthetic_masks(n_frames=2, n_tracks=2)
        labels = masks_to_sleap_labels(masks)

        # Should have 2 tracks * 2 frames = 4 masks
        assert len(labels.masks) == 4

    def test_precomputed_centroids(self):
        """Test that precomputed centroids are used correctly."""
        masks = _make_synthetic_masks(n_frames=2, n_tracks=1)
        centroids = {0: {1: (25.0, 30.0)}, 1: {1: (28.0, 30.0)}}
        labels = masks_to_sleap_labels(masks, centroids=centroids)

        # Check instance position matches precomputed centroid
        inst = labels[0].instances[0]
        assert float(inst.numpy()[0, 0]) == pytest.approx(25.0)
        assert float(inst.numpy()[0, 1]) == pytest.approx(30.0)


class TestBlendMaskOverlay:
    """Tests for _blend_mask_overlay."""

    def test_blends_correctly(self):
        """Test that mask overlay blending produces correct colors."""
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        result = _blend_mask_overlay(frame, mask, (255, 0, 0), alpha=0.5)

        # Masked region should be blend of 100 and 255
        assert result[3, 3, 0] > 100  # Red channel increased
        assert result[3, 3, 1] < 100  # Green channel decreased

        # Non-masked region unchanged
        assert result[0, 0, 0] == 100

    def test_does_not_modify_input(self):
        """Test that blending does not modify the input frame."""
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        original = frame.copy()
        mask = np.ones((10, 10), dtype=bool)
        _blend_mask_overlay(frame, mask, (255, 0, 0), alpha=0.5)
        np.testing.assert_array_equal(frame, original)


class TestRenderCtcVideo:
    """Tests for render_ctc_video."""

    def test_basic_render(self, tmp_path):
        """Test basic video rendering from mask array."""
        masks = _make_synthetic_masks(n_frames=5, n_tracks=2)
        out_path = tmp_path / "test.mp4"
        result = render_ctc_video(masks, out_path, show_progress=False, fps=10.0)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_no_raw_frames(self, tmp_path):
        """Test rendering with black background (no raw frames)."""
        masks = _make_synthetic_masks(n_frames=3, n_tracks=2)
        out_path = tmp_path / "no_raw.mp4"
        result = render_ctc_video(
            masks, out_path, raw_frames=None, show_progress=False, fps=10.0
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_with_raw_frames(self, tmp_path):
        """Test rendering with raw frame background."""
        masks = _make_synthetic_masks(n_frames=3, n_tracks=2)
        raw = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        out_path = tmp_path / "with_raw.mp4"
        result = render_ctc_video(
            masks, out_path, raw_frames=raw, show_progress=False, fps=10.0
        )
        assert result.exists()

    def test_from_tiff(self, tmp_path):
        """Test rendering from a TIFF file path."""
        masks = _make_synthetic_masks(n_frames=3, n_tracks=2)
        tiff_path = tmp_path / "masks.tif"
        tifffile.imwrite(str(tiff_path), masks)

        out_path = tmp_path / "from_tiff.mp4"
        result = render_ctc_video(
            str(tiff_path), out_path, show_progress=False, fps=10.0
        )
        assert result.exists()

    def test_disable_features(self, tmp_path):
        """Test rendering with all optional features disabled."""
        masks = _make_synthetic_masks(n_frames=3, n_tracks=2)
        out_path = tmp_path / "minimal.mp4"
        result = render_ctc_video(
            masks,
            out_path,
            show_ids=False,
            show_masks=False,
            show_trails=False,
            show_centroids=False,
            show_progress=False,
            fps=10.0,
        )
        assert result.exists()

    def test_scale(self, tmp_path):
        """Test rendering with scale factor."""
        masks = _make_synthetic_masks(n_frames=2, n_tracks=1)
        out_path = tmp_path / "scaled.mp4"
        result = render_ctc_video(
            masks, out_path, scale=2.0, show_progress=False, fps=10.0
        )
        assert result.exists()

    def test_invalid_mask_shape(self, tmp_path):
        """Test that 2D mask array raises ValueError."""
        masks = np.zeros((64, 64), dtype=np.uint16)  # 2D, not 3D
        out_path = tmp_path / "bad.mp4"
        with pytest.raises(ValueError, match="Expected 3D"):
            render_ctc_video(masks, out_path, show_progress=False)


class TestRenderSlpVideo:
    """Tests for render_slp_video."""

    def test_basic_render(self, tmp_path):
        """Test basic video rendering from Labels."""
        labels = _make_synthetic_labels(n_frames=5, n_tracks=2, n_nodes=2)
        out_path = tmp_path / "test_slp.mp4"
        result = render_slp_video(labels, out_path, show_progress=False, fps=10.0)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_from_slp_file(self, tmp_path):
        """Test rendering from a .slp file path."""
        labels = _make_synthetic_labels(n_frames=3, n_tracks=2, n_nodes=2)
        slp_path = tmp_path / "labels.slp"
        labels.save(str(slp_path))

        out_path = tmp_path / "from_slp.mp4"
        result = render_slp_video(
            str(slp_path), out_path, show_progress=False, fps=10.0
        )
        assert result.exists()

    def test_disable_features(self, tmp_path):
        """Test rendering with trails and IDs disabled."""
        labels = _make_synthetic_labels(n_frames=3, n_tracks=2, n_nodes=2)
        out_path = tmp_path / "minimal_slp.mp4"
        result = render_slp_video(
            labels,
            out_path,
            show_ids=False,
            show_trails=False,
            show_nodes=False,
            show_edges=False,
            show_progress=False,
            fps=10.0,
        )
        assert result.exists()

    def test_single_track(self, tmp_path):
        """Test rendering with a single track."""
        labels = _make_synthetic_labels(n_frames=3, n_tracks=1, n_nodes=2)
        out_path = tmp_path / "single_track.mp4"
        result = render_slp_video(labels, out_path, show_progress=False, fps=10.0)
        assert result.exists()

    def test_no_instances(self, tmp_path):
        """Test rendering labels with no instances (empty frames)."""
        skeleton = sio.Skeleton(nodes=["centroid"])
        video = _make_temp_video(3, 32, 32)
        labels = sio.Labels(
            labeled_frames=[
                sio.LabeledFrame(video=video, frame_idx=t, instances=[])
                for t in range(3)
            ],
            skeletons=[skeleton],
        )
        out_path = tmp_path / "empty.mp4"
        result = render_slp_video(labels, out_path, show_progress=False, fps=10.0)
        assert result.exists()


class TestRenderCLI:
    """Tests for the dreem render CLI command."""

    def test_render_tiff_command(self, tmp_path):
        """Test render CLI with a valid TIFF file."""
        from typer.testing import CliRunner

        from dreem.cli import app

        # Create test TIFF
        masks = _make_synthetic_masks(n_frames=3, n_tracks=2)
        tiff_path = tmp_path / "masks.tif"
        tifffile.imwrite(str(tiff_path), masks)

        out_path = tmp_path / "output.mp4"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["render", str(tiff_path), "--output", str(out_path), "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()

    def test_render_slp_command(self, tmp_path):
        """Test render CLI with a valid .slp file."""
        from typer.testing import CliRunner

        from dreem.cli import app

        labels = _make_synthetic_labels(n_frames=3, n_tracks=2, n_nodes=2)
        slp_path = tmp_path / "labels.slp"
        labels.save(str(slp_path))

        out_path = tmp_path / "output_slp.mp4"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["render", str(slp_path), "--output", str(out_path), "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()

    def test_render_unsupported_format(self, tmp_path):
        """Test render CLI with an unsupported file format."""
        from typer.testing import CliRunner

        from dreem.cli import app

        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b\n1,2\n")

        out_path = tmp_path / "output.mp4"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["render", str(bad_file), "--output", str(out_path)],
        )
        assert result.exit_code != 0

    def test_render_missing_file(self, tmp_path):
        """Test render CLI with a nonexistent file."""
        from typer.testing import CliRunner

        from dreem.cli import app

        out_path = tmp_path / "output.mp4"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "render",
                str(tmp_path / "nonexistent.tif"),
                "--output",
                str(out_path),
            ],
        )
        assert result.exit_code != 0
