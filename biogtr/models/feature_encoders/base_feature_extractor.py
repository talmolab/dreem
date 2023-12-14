"""Module containing feature extractor interface."""
from torch import nn
from torch.nn import functional as F
from torch import TensorType


class BaseFeatureExtractor(nn.Module):
    """Feature Extraction Interface.

    New feature extractors must implement a way to get features using `_forward_features`.
    Features are then projected into dimension expected for inputs into transformer and optionally normalized
    """

    def __init__(
        self, d_model: int = 512, normalize: bool = True, project: bool = True
    ) -> None:
        """Initialize feature extractor.

        Args:
            d_model: dimension feature vectors should be projected to be compatible with transformer dimensionality.
            normalize: whether or not to normalize feature vectors after projection.
            project: Whether or not to project features into new dimension.
        """
        super().__init__()

        self.out_dim = d_model
        self.normalize = normalize
        if project:
            self.out_layer = nn.LazyLinear(self.out_dim)
        else:
            self.out_layer = nn.Identity()

    def _forward_features(self, x: TensorType) -> TensorType:
        """Extract unnormalized feature vectors.

        Args:
            x: A (B, *) tensor containing the raw input to get features from.

        Returns:
            feats: A (B, d_model) tensor containing the feature vectors for each data instance in the batch.
        """
        raise NotImplementedError("Must be implemented in subclass!")

    def _project_features(self, x: TensorType) -> TensorType:
        """Map feature vectors to output dimension using linear layer.

        Args:
            x: A (B, D) tensor containing the feature vectors for each data instance in the batch.

        Returns:
            projected_feats: A (B, d_model) tensor containing the feature vectors for each data instance in the batch projected into same space as transformer inputs.
        """
        return self.out_layer(x)

    def forward(self, x: TensorType):
        """Extract features, project, then optionally normalize.

        Args:
            x: A (B, *) tensor containing the raw input to get features from.

        Returns:
            projected_feats: A (B, d_model) tensor containing the feature vectors for each data instance in the batch projected into same space as transformer inputs.
        """
        feats = self._forward_features(x)

        projected_feats = self._project_features(feats)  # (B, d_model)
        if self.normalize:
            projected_feats = F.normalize(projected_feats)
        return projected_feats
