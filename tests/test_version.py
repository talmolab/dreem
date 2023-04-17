import biogtr


def test_version():
    assert biogtr.__version__ == biogtr.version.__version__
