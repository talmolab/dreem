"""Module containing different position and temporal embeddings."""

from typing import Tuple, Optional
import math
import torch

# todo: add named tensors, clean variable names


class Embedding(torch.nn.Module):
    """Class that wraps around different embedding types.

    Used for both learned and fixed embeddings.
    """

    EMB_TYPES = {
        "temp": {},
        "pos": {"over_boxes"},
        "": {},
        None: {},
    }  # dict of valid args:keyword params
    EMB_MODES = {
        "fixed": {"temperature", "scale", "normalize"},
        "learned": {"emb_num"},
    }  # dict of valid args:keyword params

    def __init__(
        self,
        type: str,
        mode: Optional[str] = "fixed",
        features: Optional[int] = 128,
        emb_num: Optional[int] = 16,
        over_boxes: Optional[bool] = True,
        temperature: Optional[int] = 10000,
        normalize: Optional[bool] = False,
        scale: Optional[float] = None,
    ):
        """Initialize embeddings.

        Args:
            type: The type of embedding to compute. Must be one of `{"temp", "pos", ""}`
            mode: The mode or function used to map positions to vector embeddings.
            Must be one of `{"fixed", "learned"}`
            features: The embedding dimensions.
            Must match the dimension of the input vectors for the transformer model.
            emb_num: the number of embeddings in the `self.lookup` table (Only used in learned embeddings).
            over_boxes: Whether to compute the position embedding for each bbox coordinate (y1x1y2x2) or the centroid + bbox size (yxwh).
            temperature: the temperature constant to be used when computing the sinusoidal position embedding
            normalize: whether or not to normalize the positions (Only used in fixed embeddings).
            scale: factor by which to scale the positions after normalizing (Only used in fixed embeddings).
        """
        self._check_init_args(type, mode)

        super().__init__()

        self.type = type
        self.mode = mode
        self.features = features
        self.emb_num = emb_num
        self.over_boxes = over_boxes
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        if self.normalize and self.scale is None:
            self.scale = 2 * math.pi

        self._emb_func = lambda tensor: torch.zeros(
            (tensor.shape[0], self.features), dtype=tensor.dtype, device=tensor.device
        )

        if self.mode == "learned":
            if self.type == "pos":
                self.lookup = torch.nn.Embedding(self.emb_num * 4, self.features // 4)
            else:
                self.lookup = torch.nn.Embedding(self.emb_num, self.features)

            if self.type == "pos":
                self._emb_func = self._learned_pos_embedding
            elif self.type == "temp":
                self._emb_func = self._learned_temp_embedding

        elif self.mode == "fixed":
            if self.type == "pos":
                self._emb_func = self._sine_box_embedding
            elif self.type == "temp":
                pass  # TODO Implement fixed sine temporal embedding

    def _check_init_args(self, type: str, mode: str):
        """Check whether the correct arguments were passed to initialization.

        Args:
            type: The type of embedding to compute. Must be one of `{"temp", "pos", ""}`
            mode: The mode or function used to map positions to vector embeddings.
            Must be one of `{"fixed", "learned"}`

        Raises:
            ValueError:
              * if the incorrect `type` or `mode` string are passed
              * One of the kwargs needed for the `type`/`mode` embedding to be computed (See kwargs)
            NotImplementedError: if `type` is `temp` and `mode` is `fixed`.
        """
        if type.lower() not in self.EMB_TYPES:
            raise ValueError(
                f"Embedding `type` must be one of {self.EMB_TYPES} not {type}"
            )

        if mode.lower() not in self.EMB_MODES:
            raise ValueError(
                f"Embedding `mode` must be one of {self.EMB_MODE} not {mode}"
            )

        if mode == "fixed" and type == "temp":
            raise NotImplementedError("TODO: Implement Fixed Sinusoidal Temp Embedding")

    def forward(self, seq_positions: torch.Tensor) -> torch.Tensor:
        """Get the sequence positional embeddings.

        Args:
            seq_positions:
                * An `N` x 1 tensor where seq_positions[i] represents the temporal position of instance_i in the sequence.
                * An `N` x 4 tensor where seq_positions[i] represents the [y1, x1, y2, x2] spatial locations of instance_i in the sequence.

        Returns:
            An `N` x `self.features` tensor representing the corresponding spatial or temporal embedding.
        """
        return self._emb_func(seq_positions)

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

    def _sine_box_embedding(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute sine positional embeddings for boxes using given parameters.

        Args:
            boxes: the input boxes.

        Returns:
            torch.Tensor, the sine positional embeddings.
        """
        if self.scale is not None and self.normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if self.scale is None:
            self.scale = 2 * math.pi

        if len(boxes.size()) == 2:
            boxes = boxes.unsqueeze(0)

        if self.normalize:
            boxes = boxes / (boxes[:, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.features // 4, dtype=torch.float32)

        dim_t = self.temperature ** (
            2 * self._torch_int_div(dim_t, 2) / self.features // 4
        )

        # (b, n_t, 4, D//4)
        pos_emb = boxes[:, :, :, None] / dim_t.to(boxes.device)

        pos_emb = torch.stack(
            (pos_emb[:, :, :, 0::2].sin(), pos_emb[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # (n_t, D)
        pos_emb = pos_emb.squeeze(0).flatten(1)

        return pos_emb

    def _learned_pos_embedding(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute learned positional embeddings for boxes using given parameters.

        Args:
            boxes: the input boxes.

        Returns:
            torch.Tensor, the learned positional embeddings.
        """
        pos_lookup = self.lookup

        N = boxes.shape[0]
        boxes = boxes.view(N, 4)

        if self.over_boxes:
            xywh = boxes
        else:
            xywh = torch.cat(
                [(boxes[:, 2:] + boxes[:, :2]) / 2, (boxes[:, 2:] - boxes[:, :2])],
                dim=1,
            )

        left_ind, right_ind, left_weight, right_weight = self._compute_weights(xywh)

        f = pos_lookup.weight.shape[1]

        pos_emb_table = pos_lookup.weight.view(self.emb_num, 4, f)  # T x 4 x (D * 4)

        left_emb = pos_emb_table.gather(
            0, left_ind[:, :, None].to(pos_emb_table.device).expand(N, 4, f)
        )  # N x 4 x d
        right_emb = pos_emb_table.gather(
            0, right_ind[:, :, None].to(pos_emb_table.device).expand(N, 4, f)
        )  # N x 4 x d
        pos_emb = left_weight[:, :, None] * left_emb.to(
            left_weight.device
        ) + right_weight[:, :, None] * right_emb.to(right_weight.device)

        pos_emb = pos_emb.view(N, 4 * f)

        return pos_emb

    def _learned_temp_embedding(self, times: torch.Tensor) -> torch.Tensor:
        """Compute learned temporal embeddings for times using given parameters.

        Args:
            times: the input times.

        Returns:
            torch.Tensor, the learned temporal embeddings.
        """
        temp_lookup = self.lookup
        N = times.shape[0]

        left_ind, right_ind, left_weight, right_weight = self._compute_weights(times)

        left_emb = temp_lookup.weight[
            left_ind.to(temp_lookup.weight.device)
        ]  # T x D --> N x D
        right_emb = temp_lookup.weight[right_ind.to(temp_lookup.weight.device)]

        temp_emb = left_weight[:, None] * right_emb.to(
            left_weight.device
        ) + right_weight[:, None] * left_emb.to(right_weight.device)

        return temp_emb.view(N, self.features)

    def _compute_weights(self, data: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute left and right learned embedding weights.

        Args:
            data: the input data (e.g boxes or times).

        Returns:
            A torch.Tensor for each of the left/right indices and weights, respectively
        """
        data = data * self.emb_num

        left_ind = data.clamp(min=0, max=self.emb_num - 1).long()  # N x 4
        right_ind = (left_ind + 1).clamp(min=0, max=self.emb_num - 1).long()  # N x 4

        left_weight = data - left_ind.float()  # N x 4

        right_weight = 1.0 - left_weight

        return left_ind, right_ind, left_weight, right_weight
