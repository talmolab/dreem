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
        embedding_agg_method: str
    ):
        """Initialize an instance of ATTWeightHead.

        Args:
            feature_dim: The dimensionality of input features.
            num_layers: The number of hidden layers in the MLP.
            dropout: Dropout probability.
            embedding_agg_method: how the embeddings are aggregated; average/stack/concatenate
        """
        super().__init__()
        self.embedding_agg_method = embedding_agg_method

        # if using stacked embeddings, use 1x1 conv with x,y,t embeddings as channels  
        if self.embedding_agg_method == "stack":
            self.conv_1x1 = torch.nn.Conv2d(in_channels=3,out_channels=1,
                                            kernel_size=1,stride=1,padding=0)
            self.q_proj = self.conv_1x1
            self.k_proj = self.conv_1x1
        else:
            self.q_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)
            self.k_proj = MLP(feature_dim, feature_dim, feature_dim, num_layers, dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the attention weights of a query tensor using the key tensor.

        Args:
            query: Input tensor of shape (batch_size, num_frame_instances, feature_dim).
            key: Input tensor of shape (batch_size, num_window_instances, feature_dim).

        Returns:
            Output tensor of shape (batch_size, num_frame_instances, num_window_instances).
        """
        # if stacked embeddings, create channels for each x,y,t embedding dimension
        if self.embedding_agg_method == "stack":
            key = 
            query = 

        k = self.k_proj(key)
        q = self.q_proj(query)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        return attn_weights  # (B, N_t, N)
