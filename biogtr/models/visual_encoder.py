"""Module for different visual feature extractors."""

from typing import Tuple
import torch
import timm
import torch.nn.functional as F

# import timm

# todo: do we want to make timm a dependency?
# todo: add named tensor support


class VisualEncoder(torch.nn.Module):
    """Class wrapping around a visual feature extractor backbone.

    Currently CNN only.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        d_model: int = 512,
        in_chans: int = 3,
        pretrained: bool = False,
        **kwargs,
    ):
        """Initialize Visual Encoder.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            d_model (int): Output embedding dimension.
            in_chans: the number of input channels of the image.
            pretrained: whether or not to use pretrained weights from hugging_face
            kwargs: see `timm.create_model` for kwargs.
        """
        super().__init__()

        self.model_name = model_name.lower()
        self.d_model = d_model
        if in_chans == 1:
            self.in_chans = 3
        else:
            self.in_chans = in_chans

        self.feature_extractor = timm.create_model(
            model_name=self.model_name,
            in_chans=self.in_chans,
            pretrained=pretrained,
            num_classes=0,
            **kwargs,
        )

        self.out_layer = torch.nn.Linear(
            self.encoder_dim(self.feature_extractor), self.d_model
        )

    def encoder_dim(self, model: torch.nn.Module) -> int:
        """Compute dummy forward pass of encoder model and get embedding dimension.

        Args:
            model: a vision encoder model.

        Returns:
            The embedding dimension size.
        """
        _ = model.eval()
        dummy_output = model(torch.randn(1, self.in_chans, 224, 224))
        _ = model.train()  # to be safe
        return dummy_output.shape[-1]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            feats: Normalized output tensor of shape (B, d_model).
        """
        # If grayscale, tile the image to 3 channels.
        if img.shape[1] == 1:
            img = img.repeat([1, 3, 1, 1])  # (B, nc=3, H, W)
        # Extract image features
        feats = self.feature_extractor(
            img
        )  # (B, out_dim, 1, 1) if using resnet18 backbone.

        # Reshape feature vectors
        feats = feats.reshape([img.shape[0], -1])  # (B, out_dim)

        # Map feature vectors to output dimension using linear layer.
        feats = self.out_layer(feats)  # (B, d_model)

        # Normalize output feature vectors.
        feats = F.normalize(feats)  # (B, d_model)

        return feats
