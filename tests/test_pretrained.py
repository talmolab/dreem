"""Tests for pretrained model registry and download utilities."""

from unittest.mock import patch

from dreem.io.pretrained import (
    PRETRAINED_MODELS,
    is_pretrained_shortname,
    list_pretrained_models,
    resolve_checkpoint,
)


def test_registry_has_animals():
    """Test that the animals model is registered."""
    assert "animals" in PRETRAINED_MODELS
    assert (
        PRETRAINED_MODELS["animals"]["repo_id"] == "talmolab/dreem-animals-pretrained"
    )
    assert PRETRAINED_MODELS["animals"]["filename"] == "animals-pretrained.ckpt"


def test_registry_has_microscopy():
    """Test that the microscopy model is registered."""
    assert "microscopy" in PRETRAINED_MODELS
    assert (
        PRETRAINED_MODELS["microscopy"]["repo_id"]
        == "talmolab/dreem-microscopy-pretrained"
    )
    assert PRETRAINED_MODELS["microscopy"]["filename"] == "pretrained-microscopy.ckpt"


def test_is_pretrained_shortname():
    """Test shortname detection for registered and unregistered names."""
    assert is_pretrained_shortname("animals") is True
    assert is_pretrained_shortname("microscopy") is True
    assert is_pretrained_shortname("./models/my_model.ckpt") is False
    assert is_pretrained_shortname("nonexistent") is False
    assert is_pretrained_shortname("") is False


def test_list_pretrained_models():
    """Test that list_pretrained_models returns a copy of the registry."""
    models = list_pretrained_models()
    assert "animals" in models
    assert "microscopy" in models
    # Ensure it returns a copy
    models["test"] = {"repo_id": "test", "filename": "test.ckpt"}
    assert "test" not in PRETRAINED_MODELS


def test_resolve_passthrough_regular_path():
    """Test that regular paths are returned unchanged."""
    path = "./models/my_model.ckpt"
    assert resolve_checkpoint(path) == path


def test_resolve_passthrough_absolute_path():
    """Test that absolute paths are returned unchanged."""
    path = "/home/user/models/my_model.ckpt"
    assert resolve_checkpoint(path) == path


@patch("dreem.io.pretrained.hf_hub_download")
def test_resolve_animals(mock_download):
    """Test that 'animals' shortname downloads from HuggingFace."""
    mock_download.return_value = "/cache/animals-pretrained.ckpt"
    result = resolve_checkpoint("animals")
    mock_download.assert_called_once_with(
        repo_id="talmolab/dreem-animals-pretrained",
        filename="animals-pretrained.ckpt",
    )
    assert result == "/cache/animals-pretrained.ckpt"


@patch("dreem.io.pretrained.hf_hub_download")
def test_resolve_microscopy(mock_download):
    """Test that 'microscopy' shortname downloads from HuggingFace."""
    mock_download.return_value = "/cache/pretrained-microscopy.ckpt"
    result = resolve_checkpoint("microscopy")
    mock_download.assert_called_once_with(
        repo_id="talmolab/dreem-microscopy-pretrained",
        filename="pretrained-microscopy.ckpt",
    )
    assert result == "/cache/pretrained-microscopy.ckpt"
