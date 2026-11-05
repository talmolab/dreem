"""Module containing different components of multi-head attention heads."""

import torch

from dreem.models.mlp import MLP

# todo: add named tensors


class ATTWeightHead(torch.nn.Module):
    """Single attention head."""

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """Initialize an instance of ATTWeightHead.

        Args:
            feature_dim: The dimensionality of input features.
            num_layers: The number of hidden layers in the MLP.
            dropout: Dropout probability.
        """
        super().__init__()

        self.q_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)
        self.k_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)
        # Optional rotary embedding (`geometry.RotaryXYT`) applied to q and k; set by
        # `Transformer` when `embedding_meta["rope"]` is given.
        self.rope = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the attention weights of a query tensor using the key tensor.

        Args:
            query: Input tensor of shape (batch_size, num_frame_instances, feature_dim).
            key: Input tensor of shape (batch_size, num_window_instances, feature_dim).
            query_pos: (num_frame_instances, 3) (y, x, t) positions, used by `rope`.
            key_pos: (num_window_instances, 3) (y, x, t) positions, used by `rope`.

        Returns:
            Output tensor of shape
            (batch_size, num_frame_instances, num_window_instances).
        """
        k = self.k_proj(key)
        q = self.q_proj(query)
        if self.rope is not None:
            q = self.rope(q, query_pos)
            k = self.rope(k, key_pos)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        return attn_weights  # (B, N_t, N)
