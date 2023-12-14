"""Module for different visual feature extractors."""

from typing import Tuple
from biogtr.models.feature_encoders.base_feature_extractor import BaseFeatureExtractor
import torch
import torchvision


# import timm

# todo: do we want to make timm a dependency?
# todo: add named tensor support


class VisualEncoder(BaseFeatureExtractor):
    """Class wrapping around a visual feature extractor backbone.

    Currently CNN only.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        cfg: dict = {},
        d_model: int = 512,
        normalize: bool = True,
    ):
        """Initialize Visual Encoder.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            cfg (dict): Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"weights": "ResNet18_Weights.DEFAULT"}`
            d_model (int): Output embedding dimension.
            normalize: Whether to normalize after projection.
        """
        super().__init__(d_model, normalize)

        self.model_name = model_name

        self.feature_extractor = self.select_feature_extractor(model_name, cfg)

        self.feature_extractor = torch.nn.Sequential(
            *list(self.feature_extractor.children())[:-1]
        )

    def select_feature_extractor(
        self, model_name: str, cfg: dict
    ) -> Tuple[torch.nn.Module, int]:
        """Get feature extractor based on name and config.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            cfg (dict): Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"weights": "ResNet18_Weights.DEFAULT"}`

        Returns:
            The CNN feature extractor and output dimension for the given CNN architecture.
        """
        if model_name == "resnet18":
            model = torchvision.models.resnet18(**cfg)
        elif model_name == "resnet50":
            model = torchvision.models.resnet50(**cfg)
        else:
            raise ValueError(f"{model_name} model not found.")

        return model

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            feats: Normalized output tensor of shape (B, d_model).
        """
        # If grayscale, tile the image to 3 channels.
        if x.shape[1] == 1:
            x = x.repeat([1, 3, 1, 1])  # (B, nc=3, H, W)

        # Extract image features
        feats = self.feature_extractor(
            x
        )  # (B, out_dim, 1, 1) if using resnet18 backbone.

        # Reshape feature vectors
        feats = feats.reshape([x.shape[0], -1])  # (B, out_dim)

        return feats
