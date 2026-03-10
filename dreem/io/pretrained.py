"""Pretrained model registry and HuggingFace download utilities."""

from __future__ import annotations

import logging
import re

from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger("dreem.io.pretrained")

# Registry mapping shortnames to HuggingFace repo IDs and checkpoint filenames.
PRETRAINED_MODELS: dict[str, dict[str, str]] = {
    "animals": {
        "repo_id": "talmolab/dreem-animals-pretrained",
        "filename": "animals-pretrained.ckpt",
        "config": "animals-pretrained-config.yaml",
    },
    "microscopy": {
        "repo_id": "talmolab/dreem-microscopy-pretrained",
        "filename": "pretrained-microscopy.ckpt",
        "config": "microscopy-pretrained-config.yaml",
    },
}

# Pattern matching org/repo_name format (e.g. "talmolab/dreem-animals-pretrained")
_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")

# Pattern matching HuggingFace URLs
_HF_URL_PATTERN = re.compile(
    r"^https?://huggingface\.co/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"
)


def is_pretrained_shortname(checkpoint: str) -> bool:
    """Check if a checkpoint string is a pretrained model shortname.

    Args:
        checkpoint: Checkpoint path or shortname string.

    Returns:
        True if the string matches a registered pretrained model shortname.
    """
    return checkpoint in PRETRAINED_MODELS


def _is_hf_repo_id(checkpoint: str) -> bool:
    """Check if a string looks like a HuggingFace repo ID (org/repo).

    Args:
        checkpoint: String to check.

    Returns:
        True if the string matches the org/repo pattern.
    """
    return bool(_REPO_ID_PATTERN.match(checkpoint))


def _parse_hf_url(checkpoint: str) -> str | None:
    """Extract repo ID from a HuggingFace URL.

    Args:
        checkpoint: URL string to parse.

    Returns:
        The repo ID (org/repo) if the URL matches, None otherwise.
    """
    match = _HF_URL_PATTERN.match(checkpoint)
    return match.group(1) if match else None


def _find_checkpoint_in_repo(repo_id: str) -> str:
    """Find the checkpoint file in a HuggingFace repo.

    Looks for files ending in ``.ckpt`` in the repo. Raises an error if none
    or more than one are found.

    Args:
        repo_id: The HuggingFace repo ID (e.g. "talmolab/dreem-animals-pretrained").

    Returns:
        The filename of the checkpoint file.

    Raises:
        FileNotFoundError: If no ``.ckpt`` file is found in the repo.
        ValueError: If multiple ``.ckpt`` files are found.
    """
    files = list_repo_files(repo_id)
    ckpt_files = [f for f in files if f.endswith(".ckpt")]

    if len(ckpt_files) == 0:
        raise FileNotFoundError(
            f"No .ckpt file found in HuggingFace repo '{repo_id}'. "
            f"Available files: {files}"
        )
    if len(ckpt_files) > 1:
        raise ValueError(
            f"Multiple .ckpt files found in HuggingFace repo '{repo_id}': "
            f"{ckpt_files}. Please download manually and pass a local path."
        )

    return ckpt_files[0]


def _find_config_in_repo(repo_id: str) -> str | None:
    """Find a config YAML file in a HuggingFace repo.

    Looks for files ending in ``.yaml`` or ``.yml`` (excluding README-like files).

    Args:
        repo_id: The HuggingFace repo ID.

    Returns:
        The config filename if exactly one is found, None otherwise.
    """
    files = list_repo_files(repo_id)
    config_files = [
        f
        for f in files
        if (f.endswith(".yaml") or f.endswith(".yml"))
        and not f.startswith(".")
        and "config" in f.lower()
    ]

    if len(config_files) == 1:
        return config_files[0]
    return None


def _download_from_repo(repo_id: str) -> str:
    """Download a checkpoint from a HuggingFace repo.

    Also downloads the config YAML if one is found.

    Args:
        repo_id: The HuggingFace repo ID (e.g. "talmolab/dreem-animals-pretrained").

    Returns:
        The local path to the downloaded checkpoint file.
    """
    filename = _find_checkpoint_in_repo(repo_id)

    logger.info(f"Downloading checkpoint from {repo_id}/{filename}")

    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    logger.info(f"Checkpoint cached at: {local_path}")

    # Also download config if available
    config_file = _find_config_in_repo(repo_id)
    if config_file:
        config_path = hf_hub_download(repo_id=repo_id, filename=config_file)
        logger.info(f"Config cached at: {config_path}")

    return local_path


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint path, downloading from HuggingFace if needed.

    Supports three forms of HuggingFace references:

    - **Shortnames**: ``"animals"`` or ``"microscopy"`` (registered models)
    - **Repo IDs**: ``"talmolab/dreem-animals-pretrained"`` (org/repo format)
    - **URLs**: ``"https://huggingface.co/talmolab/dreem-animals-pretrained"``

    All downloads use ``huggingface_hub``, which caches files in
    ``~/.cache/huggingface/hub``. Subsequent calls skip the download.

    For registered shortnames, the config YAML is also downloaded alongside
    the checkpoint. For arbitrary repos, the config is downloaded if a single
    config YAML file is found.

    If ``checkpoint`` is none of the above, it is returned unchanged
    (assumed to be a local file path).

    Args:
        checkpoint: A pretrained model shortname, HuggingFace repo ID,
            HuggingFace URL, or a local filesystem path.

    Returns:
        The resolved local path to the checkpoint file.
    """
    checkpoint = str(checkpoint)

    # 1. Check registered shortnames
    if is_pretrained_shortname(checkpoint):
        model_info = PRETRAINED_MODELS[checkpoint]
        repo_id = model_info["repo_id"]
        filename = model_info["filename"]

        logger.info(
            f"Downloading pretrained model '{checkpoint}' from {repo_id}/{filename}"
        )

        local_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Also download the config
        if "config" in model_info:
            config_path = hf_hub_download(
                repo_id=repo_id, filename=model_info["config"]
            )
            logger.info(f"Config cached at: {config_path}")

        logger.info(f"Pretrained model cached at: {local_path}")

        return local_path

    # 2. Check HuggingFace URL
    repo_id = _parse_hf_url(checkpoint)
    if repo_id is not None:
        return _download_from_repo(repo_id)

    # 3. Check org/repo format
    if _is_hf_repo_id(checkpoint):
        return _download_from_repo(checkpoint)

    # 4. Assume local path
    return checkpoint


def resolve_config(checkpoint: str) -> str | None:
    """Resolve the config YAML path for a pretrained model.

    For registered shortnames, returns the path to the cached config YAML
    downloaded alongside the checkpoint. For arbitrary repos, attempts to
    find and download a config file.

    Returns None gracefully if no config is available or if the download fails.

    Args:
        checkpoint: A pretrained model shortname, HuggingFace repo ID,
            HuggingFace URL, or a local filesystem path.

    Returns:
        The local path to the config YAML, or None if not available.
    """
    try:
        # 1. Check registered shortnames
        if is_pretrained_shortname(checkpoint):
            model_info = PRETRAINED_MODELS[checkpoint]
            if "config" in model_info:
                return hf_hub_download(
                    repo_id=model_info["repo_id"], filename=model_info["config"]
                )
            return None

        # 2. Check HuggingFace URL
        repo_id = _parse_hf_url(checkpoint)
        if repo_id is None and _is_hf_repo_id(checkpoint):
            repo_id = checkpoint

        if repo_id is not None:
            config_file = _find_config_in_repo(repo_id)
            if config_file:
                return hf_hub_download(repo_id=repo_id, filename=config_file)

    except Exception as e:
        logger.debug(f"Could not resolve config for '{checkpoint}': {e}")

    return None


def list_pretrained_models() -> dict[str, dict[str, str]]:
    """Return the registry of available pretrained models.

    Returns:
        A copy of the pretrained model registry.
    """
    return dict(PRETRAINED_MODELS)
