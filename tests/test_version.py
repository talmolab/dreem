"""Test version."""

import dreem


def test_version():
    """Test version is available."""
    assert dreem.__version__ is not None
    assert isinstance(dreem.__version__, str)
