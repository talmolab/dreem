"""Module containing Temporal Positional Embedding."""
import torch


class TemporalEmbedding(torch.nn.Module):
    """Temporal Positional Embedding."""

    def __init__(
        self,
        features: int = 512,
        learn_emb_num: int = 16,
        temperature: int = 10000,
        scale: float = None,
        normalize: bool = False,
    ):
        """Initialize Temporal Embedding.

        Args:
            features: number of position features to use.
            learn_emb_num: Size of the dictionary of embeddings.
            temperature: frequency factor to control spread of fixed sine embed values.
                A higher temp (e.g 10000) gives a larger spread of values.
            scale: A scale factor to use if normalizing fixed sine embedding
            normalize: Whether to normalize the input before computing fixed sine embedding
        """
        super().__init__()
        self.features = features
        self.temperature = temperature
        self.scale = scale
        self.normalize = normalize
        self.learn_emb_num = learn_emb_num

    def _compute_weights(self, data: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute left and right learned embedding weights.

        Args:
            data: the input data (e.g boxes or times).

        Returns:
            A torch.Tensor for each of the left/right indices and weights, respectively
        """
        data = data * self.learn_emb_num

        left_index = data.clamp(min=0, max=self.learn_emb_num - 1).long()  # N x 4
        right_index = (
            (left_index + 1).clamp(min=0, max=self.learn_emb_num - 1).long()
        )  # N x 4

        left_weight = data - left_index.float()  # N x 4

        right_weight = 1.0 - left_weight

        return left_index, right_index, left_weight, right_weight

    def _fixed_embedding(self, times: torch.Tensor):
        """Fixed sine embedding.

        Currently not implemented.

        Args:
            times: the input times
        Returns
            torch.Tensor, the fixed sine temporal embeddings.
        """
        raise NotImplementedError(
            "Only `learned` temp embedding is currently available!"
        )

    def _learned_embedding(self, times: torch.Tensor) -> torch.Tensor:
        """Compute learned temporal embeddings for times using given parameters.

        Args:
            times: the input times.

        Returns:
            torch.Tensor, the learned temporal embeddings.
        """
        temp_lookup = torch.nn.Embedding(self.learn_emb_num, self.features)

        N = times.shape[0]

        l, r, lw, rw = self._compute_weights(times, self.learn_emb_num)

        le = temp_lookup.weight[l.to(temp_lookup.weight.device)]  # T x D --> N x D
        re = temp_lookup.weight[r.to(temp_lookup.weight.device)]

        temp_emb = lw[:, None] * re.to(lw.device) + rw[:, None] * le.to(rw.device)

        return temp_emb.view(N, self.features)

    def forward(self, times: torch.Tensor, emb_type: str = "learned"):
        """Compute temporal embedding.

        Args:
            times: the input times.
            emb_type: String representing which embedding type to compute.
            Must be one of "learned" or "fixed"/"sine" (however, currently "fixed"/"sine" are not implemented.)
        """
        if "learned" in emb_type.lower():
            return self._learned_embedding(times)
        else:
            return self._fixed_embedding(times)
