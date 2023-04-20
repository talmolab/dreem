from typing import Tuple
import torch
import torchvision
import torch.nn.functional as F

# import timm

# todo: do we want to make timm a dependency?
# todo: add named tensor support


class VisualEncoder(torch.nn.Module):
    def __init__(self, model_name: str, cfg: dict, d_model: int = 512):
        """
        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            cfg (dict): Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"weights": "ResNet18_Weights.DEFAULT"}`
            d_model (int): Output embedding dimension.
        """

        super().__init__()

        self.model_name = model_name
        self.d_model = d_model

        self.feature_extractor, out_dim = self.select_feature_extractor(model_name, cfg)

        self.feature_extractor = torch.nn.Sequential(
            *list(self.feature_extractor.children())[:-1]
        )

        self.out_layer = torch.nn.Linear(out_dim, d_model)

    def select_feature_extractor(
        self, model_name: str, cfg: dict
    ) -> Tuple[torch.nn.Module, int]:
        """
        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            cfg (dict): Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"weights": "ResNet18_Weights.DEFAULT"}`

        Returns:
            The CNN feature extractor and output dimension for the given CNN architecture.
        """

        if model_name == "resnet18":
            model = torchvision.models.resnet18(**cfg)
            out_dim = 512
        elif model_name == "resnet50":
            model = torchvision.models.resnet50(**cfg)
            out_dim = 2048
        else:
            raise ValueError(f"{model_name} model not found.")

        return model, out_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
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
