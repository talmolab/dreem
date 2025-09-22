"""Module for motion feature extractor."""

from typing import Any, Dict, Type

import numpy as np
import torch
import torch.nn.functional as F


class MotionEncoder(torch.nn.Module):
    """Class wrapping around a motion feature extractor backbone."""

    def __init__(self, **kwargs: Any | None):
        super().__init__()
        # TODO: Need to store the history of motion vectors, or at least the pose arrays to recompute motion vectors from.
        ...


def create_motion_encoder(d_model: int, **kwargs: Any | None) -> MotionEncoder:
    """Create a motion encoder."""
    return MotionEncoder(**kwargs)


def compute_motion_vectors(instances: list["Instance"]) -> list[torch.Tensor]:
    """Compute motion vectors for a list of instances.

    The list of instances is all the instances from a batch of frames. Therefore, it contains history.
    """
    motion_vectors = []
    for instance in instances:
        ...
        # TODO: Implement motion vector computation
    return motion_vectors