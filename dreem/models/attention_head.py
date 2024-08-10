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
        embedding_agg_method: str = "average"
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
        # ensures output represents ref instances by query instances
        if self.embedding_agg_method == "stack":
            self.q_proj = torch.nn.Conv1d(in_channels=3, out_channels=1,
                                          kernel_size=1, stride=1, padding=0
                                          )
            self.k_proj = torch.nn.Conv1d(in_channels=3, out_channels=1,
                                          kernel_size=1, stride=1, padding=0
                                          )
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
        batch_size, num_query_instances, feature_dim = query.size()
        num_window_instances = key.shape[1]

        # if stacked embeddings, create channels for each x,y,t embedding dimension
        # maps shape (1,192,1024) -> (1,64,3,1024)
        if self.embedding_agg_method == "stack":
            key = key.view(
                batch_size, 3, num_window_instances//3, feature_dim
            ).permute(0, 2, 1, 3).squeeze(0)
            query = query.view(
                batch_size, 3, num_query_instances//3, feature_dim
            ).permute(0, 2, 1, 3).squeeze(0)
            # key, query of shape (batch_size, num_instances, 3, feature_dim)
            k = self.k_proj(key).transpose(1, 0)
            q = self.q_proj(query).transpose(1, 0)
            # k,q of shape (batch_size, num_instances, feature_dim)
        else:
            k = self.k_proj(key)
            q = self.q_proj(query)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        return attn_weights  # (B, N_t, N)
