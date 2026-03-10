"""Pretrained model registry and HuggingFace download utilities."""

from __future__ import annotations

import logging

from huggingface_hub import hf_hub_download

logger = logging.getLogger("dreem.io.pretrained")

# Registry mapping shortnames to HuggingFace repo IDs and checkpoint filenames.
PRETRAINED_MODELS: dict[str, dict[str, str]] = {
    "animals": {
        "repo_id": "talmolab/dreem-animals-pretrained",
        "filename": "animals-pretrained.ckpt",
    },
    "microscopy": {
        "repo_id": "talmolab/dreem-microscopy-pretrained",
        "filename": "pretrained-microscopy.ckpt",
    },
}


def is_pretrained_shortname(checkpoint: str) -> bool:
    """Check if a checkpoint string is a pretrained model shortname.

    Args:
        checkpoint: Checkpoint path or shortname string.

    Returns:
        True if the string matches a registered pretrained model shortname.
    """
    return checkpoint in PRETRAINED_MODELS


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint path, downloading from HuggingFace if it's a shortname.

    If ``checkpoint`` is a registered shortname (e.g. ``"animals"`` or
    ``"microscopy"``), the corresponding model is downloaded from HuggingFace
    using ``huggingface_hub.hf_hub_download``, which caches files automatically
    in the standard HuggingFace cache directory (``~/.cache/huggingface/hub``).

    If ``checkpoint`` is not a shortname, it is returned unchanged.

    Args:
        checkpoint: A pretrained model shortname or a filesystem path.

    Returns:
        The resolved local path to the checkpoint file.

    Raises:
        ValueError: If the shortname is not in the registry.
    """
    if not is_pretrained_shortname(checkpoint):
        return checkpoint

    model_info = PRETRAINED_MODELS[checkpoint]
    repo_id = model_info["repo_id"]
    filename = model_info["filename"]

    logger.info(
        f"Downloading pretrained model '{checkpoint}' from {repo_id}/{filename}"
    )

    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    logger.info(f"Pretrained model cached at: {local_path}")

    return local_path


def list_pretrained_models() -> dict[str, dict[str, str]]:
    """Return the registry of available pretrained models.

    Returns:
        A copy of the pretrained model registry.
    """
    return dict(PRETRAINED_MODELS)
