"""Tests for dreem.utils.convert and the CLI convert command."""

import numpy as np
import pytest
import sleap_io as sio
from pathlib import Path

from dreem.utils.convert import convert_trackmate, tif2mp4, tif2npy


def test_tif2mp4(trackmate_lysosomes, tmp_path):
    """Test TIF to MP4 conversion creates a valid, non-empty MP4."""
    tif_path, _ = trackmate_lysosomes
    tif2mp4(tif_path, out_dir=str(tmp_path))

    mp4_path = tmp_path / f"{Path(tif_path).stem}.mp4"
    assert mp4_path.exists()
    assert mp4_path.stat().st_size > 0

    # Verify the video is readable and not blank
    vid = sio.Video.from_filename(str(mp4_path))
    frame = vid[0]
    assert frame.max() > 0, "MP4 frames are all black"


def test_tif2npy(trackmate_lysosomes, tmp_path):
    """Test TIF to NPY conversion creates an array with correct shape."""
    tif_path, _ = trackmate_lysosomes
    arr = tif2npy(tif_path, save=True, out_dir=str(tmp_path))

    npy_path = tmp_path / f"{Path(tif_path).stem}.npy"
    assert npy_path.exists()

    loaded = np.load(npy_path)
    assert loaded.shape == arr.shape
    assert loaded.ndim == 4
    assert loaded.shape[-1] == 1  # extra channel dim


def test_convert_trackmate(trackmate_lysosomes, tmp_path):
    """Test end-to-end TrackMate conversion produces .mp4 and .slp files."""
    tif_path, csv_path = trackmate_lysosomes

    convert_trackmate(
        label_files=[csv_path],
        vid_files=[tif_path],
        out_dir=str(tmp_path),
        to_mp4=True,
    )

    stem = Path(tif_path).stem
    mp4_path = tmp_path / f"{stem}.mp4"
    slp_path = tmp_path / f"{stem}.slp"

    assert mp4_path.exists()
    assert slp_path.exists()
    assert slp_path.stat().st_size > 0

    # Validate the .slp contents
    labels = sio.load_slp(str(slp_path))
    assert len(labels.labeled_frames) > 0
    assert len(labels.tracks) > 0

    # Check that instances have valid positions
    lf = labels.labeled_frames[0]
    assert len(lf.instances) > 0
    for inst in lf.instances:
        assert inst.numpy().shape == (1, 2)
        assert not np.isnan(inst.numpy()).all()


def test_convert_trackmate_missing_files(tmp_path):
    """Test that convert_trackmate raises ValueError when files are missing."""
    with pytest.raises(ValueError):
        convert_trackmate(
            label_files=[],
            vid_files=[],
            out_dir=str(tmp_path),
        )


def test_convert_cli(trackmate_lysosomes, tmp_path):
    """Test the CLI convert command end-to-end via CliRunner."""
    from typer.testing import CliRunner
    from dreem.cli import app

    tif_path, csv_path = trackmate_lysosomes
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "convert",
            "trackmate",
            "-l",
            csv_path,
            "-v",
            tif_path,
            "--output",
            str(tmp_path),
            "--to-mp4",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    stem = Path(tif_path).stem
    assert (tmp_path / f"{stem}.mp4").exists()
    assert (tmp_path / f"{stem}.slp").exists()


def test_convert_cli_unknown_format(tmp_path):
    """Test the CLI rejects unknown conversion formats."""
    from typer.testing import CliRunner
    from dreem.cli import app

    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "convert",
            "unknown_format",
            "-l",
            "dummy.csv",
            "-v",
            "dummy.tif",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "Unknown format" in result.output
