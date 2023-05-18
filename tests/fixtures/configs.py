import os
import pytest


@pytest.fixture
def config_dir(pytestconfig):
    """Dir path to sleap data."""
    return os.path.join(pytestconfig.rootdir, "tests/configs")


@pytest.fixture
def base_config(config_dir):
    return os.path.join(config_dir, "base.yaml")


@pytest.fixture
def params_config(config_dir):
    return os.path.join(config_dir, "params.yaml")
