"""Module containing spatial positional embedding."""
import torch


class SpatialEmbedding(torch.nn.Module):
    """Spatial positional embedding."""

    def __init__(
        self,
        features: int = 512,
        temperature: int = 10000,
        scale: float = None,
        normalize: bool = False,
        learn_emb_num: int = 16,
        over_boxes: bool = True,
    ):
        """Initialize Spatial Embedding.

        Args:
            features: number of position features to use.
            temperature: frequency factor to control spread of fixed sine pos embed values.
                A higher temp (e.g 10000) gives a larger spread of values
            scale: A scale factor to use if normalizing fixed sine embedding
            normalize: Whether to normalize the input before computing fixed sine embedding
            learn_emb_num: Size of the dictionary of learned embeddings.
            over_boxes: If True, use box dimensions, rather than box offset and shape when learned.
        """
        super().__init__()
        self.features = features
        self.temperature = temperature
        self.scale = scale
        self.normalize = normalize
        self.learn_emb_num = learn_emb_num
        self.over_boxes = over_boxes

    def _torch_int_div(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor
    ) -> torch.Tensor:
        """Perform integer division of two tensors.

        Args:
            tensor1: dividend tensor.
            tensor2: divisor tensor.

        Returns:
            torch.Tensor, resulting tensor.
        """
        return torch.div(tensor1, tensor2, rounding_mode="floor")

    def _fixed_embedding(
        self,
        boxes,
    ) -> torch.Tensor:
        """Compute sine positional embeddings for boxes using given parameters.

        Args:
            boxes: the input boxes.

        Returns:
            torch.Tensor, the sine positional embeddings.
        """
        # update default parameters with kwargs if available

        if self.scale is not None and self.normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if self.scale is None:
            self.scale = 2 * math.pi

        if len(boxes.size()) == 2:
            boxes = boxes.unsqueeze(0)

        if self.normalize:
            boxes = boxes / (boxes[:, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.features, dtype=torch.float32)

        dim_t = self.temperature ** (2 * self._torch_int_div(dim_t, 2) / self.features)

        # (b, n_t, 4, D//4)
        pos_emb = boxes[:, :, :, None] / dim_t.to(boxes.device)

        pos_emb = torch.stack(
            (pos_emb[:, :, :, 0::2].sin(), pos_emb[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # (n_t, D)
        pos_emb = pos_emb.squeeze(0).flatten(1)

        return pos_emb

    def _learned_embedding(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute learned positional embeddings for boxes using given parameters.

        Args:
            boxes: the input boxes.

        Returns:
            torch.Tensor, the learned positional embeddings.
        """
        pos_lookup = torch.nn.Embedding(self.learn_emb_num * 4, self.features // 4)

        N = boxes.shape[0]
        boxes = boxes.view(N, 4)

        if self.over_boxes:
            xywh = boxes
        else:
            xywh = torch.cat(
                [(boxes[:, 2:] + boxes[:, :2]) / 2, (boxes[:, 2:] - boxes[:, :2])],
                dim=1,
            )

        l, r, lw, rw = self._compute_weights(xywh)

        f = pos_lookup.weight.shape[1]

        pos_emb_table = pos_lookup.weight.view(
            self.learn_emb_num, 4, f
        )  # T x 4 x (D * 4)

        pos_le = pos_emb_table.gather(
            0, l[:, :, None].to(pos_emb_table.device).expand(N, 4, f)
        )  # N x 4 x d
        pos_re = pos_emb_table.gather(
            0, r[:, :, None].to(pos_emb_table.device).expand(N, 4, f)
        )  # N x 4 x d
        pos_emb = lw[:, :, None] * pos_re.to(lw.device) + rw[:, :, None] * pos_le.to(
            rw.device
        )

        pos_emb = pos_emb.view(N, 4 * f)

        return pos_emb

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

    def forward(self, boxes: torch.Tensor, emb_type: str = "learned") -> torch.Tensor:
        """Compute spatial positional embedding.

        Args:
            boxes: An (n, 4) tensor containing bbox coordinates
            emb_type: str representing which type of embedding to compute.
            Either "learned" or "fixed"/"sine"
        """
        if "learn" in emb_type.lower():
            return self._learned_embedding(boxes)
        else:
            return self._fixed_embedding(boxes)
