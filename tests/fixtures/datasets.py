import os
import pytest


@pytest.fixture
def data_dir(pytestconfig):
    return os.path.join(pytestconfig.rootdir, "data")


@pytest.fixture
def single_fly(data_dir):
    slp_file = os.path.join(data_dir, "single_fly.slp")
    video_file = os.path.join(data_dir, "single_fly.mp4")
    return [slp_file, video_file]


@pytest.fixture
def two_flies(data_dir):
    slp_file = os.path.join(data_dir, "two_flies.slp")
    video_file = os.path.join(data_dir, "two_flies.mp4")
    return [slp_file, video_file]


@pytest.fixture
def three_flies(data_dir):
    slp_file = os.path.join(data_dir, "three_flies.slp")
    video_file = os.path.join(data_dir, "three_flies.mp4")
    return [slp_file, video_file]


@pytest.fixture
def ten_zfish(data_dir):
    slp_file = os.path.join(data_dir, "ten_zfish.slp")
    video_file = os.path.join(data_dir, "ten_zfish.mp4")
    return [slp_file, video_file]
