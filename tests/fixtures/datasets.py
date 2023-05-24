"""Fixtures for testing biogtr."""
import glob
import os
import pytest


@pytest.fixture
def sleap_data_dir(pytestconfig):
    """Dir path to sleap data."""
    return os.path.join(pytestconfig.rootdir, "tests/data/sleap")


@pytest.fixture
def icy_data_dir(pytestconfig):
    """Dir path to icy data."""
    return os.path.join(pytestconfig.rootdir, "tests/data/microscopy/icy")


@pytest.fixture
def isbi_data_dir(pytestconfig):
    """Dir path to isbi data."""
    return os.path.join(pytestconfig.rootdir, "tests/data/microscopy/isbi")


@pytest.fixture
def trackmate_data_dir(pytestconfig):
    """Dir path to trackmate data."""
    return os.path.join(pytestconfig.rootdir, "tests/data/microscopy/trackmate")


@pytest.fixture
def single_fly(sleap_data_dir):
    """Sleap single fly .slp and video file paths."""
    slp_file = os.path.join(sleap_data_dir, "single_fly.slp")
    video_file = os.path.join(sleap_data_dir, "single_fly.mp4")
    return [slp_file, video_file]


@pytest.fixture
def two_flies(sleap_data_dir):
    """Sleap two flies .slp and video file paths."""
    slp_file = os.path.join(sleap_data_dir, "two_flies.slp")
    video_file = os.path.join(sleap_data_dir, "two_flies.mp4")
    return [slp_file, video_file]


@pytest.fixture
def three_flies(sleap_data_dir):
    """Sleap three flies .slp and video file paths."""
    slp_file = os.path.join(sleap_data_dir, "three_flies.slp")
    video_file = os.path.join(sleap_data_dir, "three_flies.mp4")
    return [slp_file, video_file]


@pytest.fixture
def ten_zfish(sleap_data_dir):
    """Idtracker 10 zebrafish slp and video file paths."""
    slp_file = os.path.join(sleap_data_dir, "ten_zfish.slp")
    video_file = os.path.join(sleap_data_dir, "ten_zfish.mp4")
    return [slp_file, video_file]


@pytest.fixture
def ten_icy_particles(icy_data_dir):
    """ICY 10 particles tif and gt xml file paths."""
    image = os.path.join(icy_data_dir, "10_cells_1_crop.tif")
    gt = os.path.join(icy_data_dir, "10_cells_1_gt.xml")
    return [image, gt]


@pytest.fixture
def isbi_microtubules(isbi_data_dir):
    image = sorted(glob.glob(os.path.join(isbi_data_dir, "microtubules", "*.tif")))
    gt = glob.glob(os.path.join(isbi_data_dir, "microtubules", "*.xml"))[0]
    return [image, gt]


@pytest.fixture
def isbi_receptors(isbi_data_dir):
    image = sorted(glob.glob(os.path.join(isbi_data_dir, "receptors", "*.tif")))
    gt = glob.glob(os.path.join(isbi_data_dir, "receptors", "*.xml"))[0]
    return [image, gt]


@pytest.fixture
def trackmate_lysosomes(trackmate_data_dir):
    image = glob.glob(os.path.join(trackmate_data_dir, "*.tif"))[0]
    gt = glob.glob(os.path.join(trackmate_data_dir, "*.csv"))[0]
    return [image, gt]
