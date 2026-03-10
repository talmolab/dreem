"""Tests for pretrained model registry and download utilities."""

from unittest.mock import patch

from dreem.io.pretrained import (
    PRETRAINED_MODELS,
    _find_checkpoint_in_repo,
    _find_config_in_repo,
    _is_hf_repo_id,
    _parse_hf_url,
    is_pretrained_shortname,
    list_pretrained_models,
    resolve_checkpoint,
    resolve_config,
)


def test_registry_has_animals():
    """Test that the animals model is registered."""
    assert "animals" in PRETRAINED_MODELS
    assert (
        PRETRAINED_MODELS["animals"]["repo_id"] == "talmolab/dreem-animals-pretrained"
    )
    assert PRETRAINED_MODELS["animals"]["filename"] == "animals-pretrained.ckpt"
    assert PRETRAINED_MODELS["animals"]["config"] == "animals-pretrained-config.yaml"


def test_registry_has_microscopy():
    """Test that the microscopy model is registered."""
    assert "microscopy" in PRETRAINED_MODELS
    assert (
        PRETRAINED_MODELS["microscopy"]["repo_id"]
        == "talmolab/dreem-microscopy-pretrained"
    )
    assert PRETRAINED_MODELS["microscopy"]["filename"] == "pretrained-microscopy.ckpt"
    assert (
        PRETRAINED_MODELS["microscopy"]["config"] == "microscopy-pretrained-config.yaml"
    )


def test_is_pretrained_shortname():
    """Test shortname detection for registered and unregistered names."""
    assert is_pretrained_shortname("animals") is True
    assert is_pretrained_shortname("microscopy") is True
    assert is_pretrained_shortname("./models/my_model.ckpt") is False
    assert is_pretrained_shortname("nonexistent") is False
    assert is_pretrained_shortname("") is False


def test_is_hf_repo_id():
    """Test HuggingFace repo ID pattern matching."""
    assert _is_hf_repo_id("talmolab/dreem-animals-pretrained") is True
    assert _is_hf_repo_id("org/repo") is True
    assert _is_hf_repo_id("my-org/my-repo.v2") is True
    assert _is_hf_repo_id("animals") is False
    assert _is_hf_repo_id("./models/model.ckpt") is False
    assert _is_hf_repo_id("https://huggingface.co/org/repo") is False
    assert _is_hf_repo_id("") is False


def test_parse_hf_url():
    """Test HuggingFace URL parsing."""
    assert (
        _parse_hf_url("https://huggingface.co/talmolab/dreem-animals-pretrained")
        == "talmolab/dreem-animals-pretrained"
    )
    assert _parse_hf_url("http://huggingface.co/org/repo") == "org/repo"
    assert (
        _parse_hf_url(
            "https://huggingface.co/talmolab/dreem-animals-pretrained/blob/main/file"
        )
        == "talmolab/dreem-animals-pretrained"
    )
    assert _parse_hf_url("./models/model.ckpt") is None
    assert _parse_hf_url("animals") is None
    assert _parse_hf_url("https://example.com/org/repo") is None


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
    """Test that 'animals' shortname downloads checkpoint and config."""
    mock_download.return_value = "/cache/animals-pretrained.ckpt"
    result = resolve_checkpoint("animals")
    assert result == "/cache/animals-pretrained.ckpt"
    # Should download both checkpoint and config
    assert mock_download.call_count == 2
    mock_download.assert_any_call(
        repo_id="talmolab/dreem-animals-pretrained",
        filename="animals-pretrained.ckpt",
    )
    mock_download.assert_any_call(
        repo_id="talmolab/dreem-animals-pretrained",
        filename="animals-pretrained-config.yaml",
    )


@patch("dreem.io.pretrained.hf_hub_download")
def test_resolve_microscopy(mock_download):
    """Test that 'microscopy' shortname downloads checkpoint and config."""
    mock_download.return_value = "/cache/pretrained-microscopy.ckpt"
    result = resolve_checkpoint("microscopy")
    assert result == "/cache/pretrained-microscopy.ckpt"
    assert mock_download.call_count == 2


@patch("dreem.io.pretrained.list_repo_files")
@patch("dreem.io.pretrained.hf_hub_download")
def test_resolve_repo_id(mock_download, mock_list_files):
    """Test that org/repo format downloads from HuggingFace."""
    mock_list_files.return_value = ["README.md", "model.ckpt", "config.yaml"]
    mock_download.return_value = "/cache/model.ckpt"
    result = resolve_checkpoint("someorg/somerepo")
    assert result == "/cache/model.ckpt"
    mock_download.assert_any_call(repo_id="someorg/somerepo", filename="model.ckpt")


@patch("dreem.io.pretrained.list_repo_files")
@patch("dreem.io.pretrained.hf_hub_download")
def test_resolve_hf_url(mock_download, mock_list_files):
    """Test that HuggingFace URLs download from the repo."""
    mock_list_files.return_value = ["README.md", "model.ckpt"]
    mock_download.return_value = "/cache/model.ckpt"
    result = resolve_checkpoint("https://huggingface.co/someorg/somerepo")
    assert result == "/cache/model.ckpt"
    mock_download.assert_called_with(repo_id="someorg/somerepo", filename="model.ckpt")


@patch("dreem.io.pretrained.list_repo_files")
def test_find_checkpoint_no_ckpt(mock_list_files):
    """Test error when no .ckpt file in repo."""
    mock_list_files.return_value = ["README.md", "config.yaml"]
    try:
        _find_checkpoint_in_repo("org/repo")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


@patch("dreem.io.pretrained.list_repo_files")
def test_find_checkpoint_multiple_ckpt(mock_list_files):
    """Test error when multiple .ckpt files in repo."""
    mock_list_files.return_value = ["model1.ckpt", "model2.ckpt"]
    try:
        _find_checkpoint_in_repo("org/repo")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


@patch("dreem.io.pretrained.list_repo_files")
def test_find_config_in_repo(mock_list_files):
    """Test finding config files in repo."""
    mock_list_files.return_value = ["README.md", "model.ckpt", "my-config.yaml"]
    assert _find_config_in_repo("org/repo") == "my-config.yaml"


@patch("dreem.io.pretrained.list_repo_files")
def test_find_config_in_repo_none(mock_list_files):
    """Test returning None when no config found."""
    mock_list_files.return_value = ["README.md", "model.ckpt"]
    assert _find_config_in_repo("org/repo") is None


@patch("dreem.io.pretrained.hf_hub_download")
def test_resolve_config_shortname(mock_download):
    """Test resolving config for a shortname."""
    mock_download.return_value = "/cache/animals-pretrained-config.yaml"
    result = resolve_config("animals")
    assert result == "/cache/animals-pretrained-config.yaml"
    mock_download.assert_called_once_with(
        repo_id="talmolab/dreem-animals-pretrained",
        filename="animals-pretrained-config.yaml",
    )


def test_resolve_config_local_path():
    """Test that resolve_config returns None for local paths."""
    assert resolve_config("./models/model.ckpt") is None


@patch("dreem.io.pretrained.hf_hub_download", side_effect=Exception("network error"))
def test_resolve_config_error_returns_none(mock_download):
    """Test that resolve_config returns None on errors."""
    result = resolve_config("animals")
    assert result is None
