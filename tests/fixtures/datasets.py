"""Fixtures for testing dreem."""

import glob
import os
from pathlib import Path

import pytest


@pytest.fixture
def sleap_data_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/data/sleap"


@pytest.fixture
def icy_data_dir(pytestconfig):
    """Dir path to icy data."""
    return Path(pytestconfig.rootdir) / "tests/data/microscopy/icy"


@pytest.fixture
def isbi_data_dir(pytestconfig):
    """Dir path to isbi data."""
    return Path(pytestconfig.rootdir) / "tests/data/microscopy/isbi"


@pytest.fixture
def trackmate_data_dir(pytestconfig):
    """Dir path to trackmate data."""
    return Path(pytestconfig.rootdir) / "tests/data/microscopy/trackmate"


@pytest.fixture
def cell_tracking_data_dir(pytestconfig):
    """Dir path to cell tracking challenge data."""
    return Path(pytestconfig.rootdir) / "tests/data/microscopy/cell_tracking"


@pytest.fixture
def single_fly(sleap_data_dir):
    """Sleap single fly .slp and video file paths."""
    slp_file = Path(sleap_data_dir) / "single_fly.slp"
    video_file = Path(sleap_data_dir) / "single_fly.mp4"
    return [slp_file, video_file]


@pytest.fixture
def two_flies(sleap_data_dir):
    """Sleap two flies .slp and video file paths."""
    slp_file = Path(sleap_data_dir) / "two_flies.slp"
    video_file = Path(sleap_data_dir) / "two_flies.mp4"
    return [slp_file, video_file]


@pytest.fixture
def two_flies_overlapping(sleap_data_dir):
    """Sleap two flies .slp and video file paths with overlapping instances."""
    slp_file = Path(sleap_data_dir) / "two_flies_noisy_detections.slp"
    video_file = Path(sleap_data_dir) / "two_flies.mp4"
    return [slp_file, video_file]


@pytest.fixture
def three_flies(sleap_data_dir):
    """Sleap three flies .slp and video file paths."""
    slp_file = Path(sleap_data_dir) / "three_flies.slp"
    video_file = Path(sleap_data_dir) / "three_flies.mp4"
    return [slp_file, video_file]


@pytest.fixture
def ten_zfish(sleap_data_dir):
    """Idtracker 10 zebrafish slp and video file paths."""
    slp_file = Path(sleap_data_dir) / "ten_zfish.slp"
    video_file = Path(sleap_data_dir) / "ten_zfish.mp4"
    return [slp_file, video_file]


@pytest.fixture
def ten_icy_particles(icy_data_dir):
    """ICY 10 particles tif and gt xml file paths."""
    image = Path(icy_data_dir) / "10_cells_1_crop.tif"
    gt = Path(icy_data_dir) / "10_cells_1_gt.xml"
    return [str(image), str(gt)]


@pytest.fixture
def isbi_microtubules(isbi_data_dir):
    """ISBI microtubuules tif and gt xml file paths."""
    isbi_micro_dir = Path(isbi_data_dir) / "microtubules"
    image = sorted(isbi_micro_dir.glob("*.tif"))
    gt = list(isbi_micro_dir.glob("*.xml"))[0]
    return [image, gt]


@pytest.fixture
def isbi_receptors(isbi_data_dir):
    """ISBI receptors tif and gt xml file paths."""
    isbi_receptor_dir = Path(isbi_data_dir) / "receptors"
    image = sorted(isbi_receptor_dir.glob("*.tif"))
    gt = list(isbi_receptor_dir.glob("*.xml"))[0]
    return [image, gt]


@pytest.fixture
def trackmate_lysosomes(trackmate_data_dir):
    """Trackmate lysosomes tif and gt csv file paths."""
    image = list(Path(trackmate_data_dir).glob("*.tif"))[0]
    gt = list(Path(trackmate_data_dir).glob("*.csv"))[0]
    return [str(image), str(gt)]


@pytest.fixture
def cell_tracking(cell_tracking_data_dir):
    """Cell tracking challenge tif and gt txt file paths."""
    gt_list = []
    raw_img_list = []
    gt_path = Path(cell_tracking_data_dir) / "test_0_GT/TRA"
    raw_img_path = Path(cell_tracking_data_dir) / "test_0"
    gt_list.append(glob.glob(os.path.join(gt_path, "*.tif")))
    # get filepaths for all tif files in raw_img_path
    raw_img_list.append(glob.glob(os.path.join(raw_img_path, "*.tif")))
    man_track_file = glob.glob(os.path.join(gt_path, "man_track.txt"))

    return (raw_img_list, gt_list, man_track_file, str(cell_tracking_data_dir))
