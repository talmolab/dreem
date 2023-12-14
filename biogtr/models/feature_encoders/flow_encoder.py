"""Module for different visual feature extractors."""

from typing import Tuple
from biogtr.models.feature_encoders.base_feature_extractor import BaseFeatureExtractor
import torch

import timm

# todo: add named tensor support


class FlowEncoder(BaseFeatureExtractor):
    """Class wrapping around a optical flow feature extractor backbone.

    Currently CNN only.
    """

    def __init__(
        self,
        cfg: dict = {"model_name": "resnet50", "pretrained": False},
        d_model: int = 512,
        normalize: bool = True,
    ):
        """Initialize Flow Encoder.

        Args:
            cfg (dict): Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"model_name": "resnet50", "pretrained": False}`
            d_model (int): Output embedding dimension.
            normalize: Whether to normalize after projection.
        """
        super().__init__(d_model, normalize)

        self.cfg = cfg

        self.feature_extractor = self.select_feature_extractor(self.cfg)

    def select_feature_extractor(self, cfg: dict) -> Tuple[torch.nn.Module, int]:
        """Get feature extractor based on name and config.

        Args:
            cfg (dict): Dictionary of arguments to pass to `timm.create_model`,
                e.g: `cfg = {"model_name"="resnet50", "pretrained"=False}`

        Returns:
            The CNN feature extractor and output dimension for the given CNN architecture.
        """
        return timm.create_model(in_chans=2, num_classes=0, **cfg)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector.

        Args:
            x: Input optical flow tensor of shape (B, 2, H, W).

        Returns:
            feats: Output tensor of shape (B, d_model).
        """
        feats = self.feature_extractor(
            x
        )  # (B, out_dim, 1, 1) if using resnet18 backbone.

        # Reshape feature vectors
        feats = feats.reshape([x.shape[0], -1])  # (B, out_dim)

        return feats
