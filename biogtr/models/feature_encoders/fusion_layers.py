"""Module containing layer for fusing multimodal embeddings."""

import torch
from biogtr.models.feature_encoders.base_feature_extractor import BaseFeatureExtractor


class Cat(BaseFeatureExtractor):
    """Concatenation Layer."""

    def __init__(self, d_model: int = 512, normalize: bool = True):
        """Initialize concatenation layer.

        Args:
            d_model: dimension to map back to after concatenation
            normalize: whether or not to normalize after projection.
        """
        super().__init__(d_model, normalize)

    def _forward_features(self, x: list[torch.TensorType]):
        """Concatenate features together.

        Args:
            x: a list of tensors containing different feature embeddings each with shape (B, d_model)

        Returns:
            Every tensor in `x` concatenated into a single torch tensor of shape `(B, sum([x_i.shape[-1] for x_i in x]))`
        """
        return torch.cat(x, dim=-1)


class Sum(BaseFeatureExtractor):
    """Vector Summation Layer."""

    def __init__(self, d_model: int = 512, normalize: bool = True):
        """Initialize summation layer.

        Args:
            d_model: Dimension to map to after summation.
            normalize: Whether or not to normalize after summation.
        """
        super().__init__(d_model, normalize, project=False)

    def _forward_features(self, x: list[torch.TensorType]):
        """Sum features together.

        Args:
            x: a list of tensors containing different feature embeddings each with shape (B, d_model)

        Returns:
            The sum of all vectors in x
        """
        x = torch.stack(x, axis=-1)
        return x.sum(dim=-1)
