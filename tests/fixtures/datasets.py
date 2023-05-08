import os
import pytest


@pytest.fixture
def sleap_data_dir(pytestconfig):
    return os.path.join(pytestconfig.rootdir, "tests/data/sleap")


@pytest.fixture
def icy_data_dir(pytestconfig):
    return os.path.join(pytestconfig.rootdir, "tests/data/microscopy/icy")


@pytest.fixture
def isbi_data_dir(pytestconfig):
    return os.path.join(pytestconfig.rootdir, "tests/data/microscopy/isbi")


@pytest.fixture
def trackmate_data_dir(pytestconfig):
    return os.path.join(pytestconfig.rootdir, "tests/data/microscopy/trackmate")


@pytest.fixture
def single_fly(sleap_data_dir):
    slp_file = os.path.join(sleap_data_dir, "single_fly.slp")
    video_file = os.path.join(sleap_data_dir, "single_fly.mp4")
    return [slp_file, video_file]


@pytest.fixture
def two_flies(sleap_data_dir):
    slp_file = os.path.join(sleap_data_dir, "two_flies.slp")
    video_file = os.path.join(sleap_data_dir, "two_flies.mp4")
    return [slp_file, video_file]


@pytest.fixture
def three_flies(sleap_data_dir):
    slp_file = os.path.join(sleap_data_dir, "three_flies.slp")
    video_file = os.path.join(sleap_data_dir, "three_flies.mp4")
    return [slp_file, video_file]


@pytest.fixture
def ten_zfish(sleap_data_dir):
    slp_file = os.path.join(sleap_data_dir, "ten_zfish.slp")
    video_file = os.path.join(sleap_data_dir, "ten_zfish.mp4")
    return [slp_file, video_file]


@pytest.fixture
def ten_icy_particles(icy_data_dir):
    image = os.path.join(icy_data_dir, "10_cells_1_crop.tif")
    gt = os.path.join(icy_data_dir, "10_cells_1_gt.xml")
    return [image, gt]
