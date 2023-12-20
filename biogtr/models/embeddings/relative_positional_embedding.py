"""Module containing Relative Positional Embeddings."""
import torch


class RelativePositionalMask(torch.nn.Module):
    """Relative Positional Mask/Embedding."""

    def __init__(
        self,
        n_head: int = 8,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 32,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Initialize Relative Positional Mask.

        Args:
            n_head: number of heads used in transformer MHA.
            cutoff_spatial:  min/max spatial bin values
            cutoff_temporal: min/max temporal_bin values
            n_spatial: number of spatial bins
            n_temporal: number of temporal bins
        """
        super().__init__()
        self._spatial_bins = self._bin_init_exp(cutoff_spatial, n_spatial)
        self._temporal_bins = self._bin_init_linear(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = torch.nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )
        self.n_head = n_head

    def _compute_positional_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute positional bias based on input coordinates.

        This function calculates the positional bias by considering both temporal and spatial distances
        between positions. Temporal distances are computed based on the frame information, and spatial
        distances are computed using the pairwise distances between positions. The distances are then
        discretized into bins, and the corresponding indices are used to select elements from a bias tensor.
        The final output is a tensor representing the positional bias for each head in a multi-head attention mechanism.

        Args:
            coords (torch.Tensor): Input tensor of shape (B, N, D), where B is the batch size,
                                N is the number of instances, and D is the spatiotemporal dimensions.
                                coords[:,0] is the temporal coordinate, while coords[:,1:] is the spatial (ie centroids, poses, masks etc).

        Returns:
            torch.Tensor: Computed positional bias tensor of shape (N, nH, N), where N is the number
                        of instances, nH is the number of attention heads.
        """
        frames = coords[..., 0]
        pos = coords[..., 1:]

        temporal_dist = frames.unsqueeze(-1) - frames.unsqueeze(-2)
        spatial_dist = torch.cdist(pos, pos)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        # do some index gymnastics such that backward is not super slow
        # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view(spatial_idx.shape + (self.n_head,))
        # -> B, nH, N, N

        bias = bias.permute(2, 1, 0)

        return bias

    def forward(
        self,
        coords: torch.Tensor,
        mode="self",
        padding_mask: torch.Tensor = None,
        query_inds=None,
    ):
        """Generate an attention mask for self-attention or cross-attention.

        This function generates an attention mask for self-attention or cross-attention based on the
        provided input coordinates. The attention mask is computed by adding a positional bias, which is
        determined by the relative spatial and temporal distances between positions. The attention mask is
        optionally masked for padding tokens if a padding mask is provided. If query indices are specified,
        the attention is computed only between the specified queries and the rest of the positions.

        Args:
            coords (torch.Tensor): Input tensor of shape (B, N, D), where B is the batch size,
                                N is the number of instances, and D is the spatiotemporal dimensions.
                                coords[:,0] is the temporal coordinate, while coords[:,1:] is the spatial (ie centroids, poses, masks etc).
            mode (str, optional): Attention mode. If "self", self-attention is performed; if "cross",
                                cross-attention is performed. Defaults to "self".
            padding_mask (torch.Tensor, optional): Mask indicating padding positions. If provided,
                                                attention is masked for padding tokens. Defaults to None.
            query_inds (List[int], optional): List of indices specifying query positions. If provided,
                                            attention is computed only between these queries and the rest.
                                            Defaults to None.

        Returns:
            torch.Tensor: Computed attention mask tensor of shape (n_head, n_query, n_nonquery),
                        where n_head is the number of attention heads, n_query is the number of query
                        positions, and n_nonquery is the number of non-query positions.
        """
        N, D = coords.shape

        if query_inds is not None:
            n_query = len(query_inds)
        else:
            n_query = N

        if mode.lower() == "self":
            #print("Using self attention mask")
            n_nonquery = n_query
        else:
            n_nonquery = N

        # dont add positional bias to self-attention if coords is None
        if coords is not None:
            pos_bias = self._compute_positional_bias(coords)

        else:
            pos_bias = torch.zeros((self.n_head, n_query, n_nonquery))
        
        attn_mask = torch.zeros((self.n_head, n_query, n_nonquery), device=pos_bias.device)

        if query_inds is not None:
            if mode.lower() == "self":
                #print("Slicing both query and nonquery")
                pos_bias = pos_bias[:, query_inds, :][:, :, query_inds]
            else:
                pos_bias = pos_bias[:, query_inds, :]

        attn_mask = attn_mask + pos_bias
        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            # add small value but not too small to keep mixed precision loss from becoming nan
            attn_mask = attn_mask.masked_fill(ignore_mask, -1e3)

        return attn_mask

    def _bin_init_exp(self, cutoff: float, n: int):
        """Initialize bins exponentially spaced up to a specified cutoff.

        This function initializes bins with exponentially increasing values up to a specified cutoff.
        The exponential spacing is achieved by taking the logarithm of the cutoff, linearly spacing
        the values from 0 to the logarithm, and exponentiating the result. The number of bins is
        determined by the parameter 'n'.

        Args:
            cutoff (float): The maximum value for the bins.
            n (int): The number of bins to generate.

        Returns:
            torch.Tensor: Exponentially spaced bins as a 1-dimensional tensor.
        """
        return torch.exp(torch.linspace(0, torch.log(torch.tensor(cutoff + 1)), n))

    def _bin_init_linear(self, cutoff: float, n: int) -> torch.Tensor:
        """Initialize bins with linear spacing within a specified range.

        This function initializes bins with linearly spaced values within a specified range.
        The range is determined by the parameter 'cutoff', which represents half the width of
        the range. The number of bins is determined by the parameter 'n'.

        Args:
            cutoff (float): The half-width of the range for linear spacing.
            n (int): The number of bins to generate.

        Returns:
            torch.Tensor: Linearly spaced bins as a 1-dimensional tensor.
        """
        return torch.linspace(-cutoff, cutoff, n)
