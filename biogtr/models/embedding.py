from typing import Tuple
import math
import torch
import torch.nn.functional as F

# todo: add named tensors, clean variable names


class Embedding(torch.nn.Module):
    def __init__(self) -> None:
        # empty init for flexibility
        pass

    def _torch_int_div(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs integer division of two tensors.
        Args:
            tensor1: dividend tensor.
            tensor2: divisor tensor.
        Returns:
            torch.Tensor, resulting tensor.
        """
        return torch.div(tensor1, tensor2, rounding_mode="floor")

    def _sine_box_embedding(
        self,
        boxes,
        embedding_dim: int = 512,
        temperature: int = 10000,
        device: str = "cpu",
        scale: float = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Generates sine positional embeddings for boxes using given parameters.
        Args:
            boxes: the input boxes.
            embedding_dim: number of position features to use.
            temperature: frequency factor to control spread of pos embed values.
                A higher temp (e.g 10000) gives a larger spread of values
            device: the device to be used (e.g., "cuda", "cpu").
            scale: A scale factor to use if normalizing
            normalize: Whether to normalize the input before computing embedding
        Returns:
            torch.Tensor, the sine positional embeddings.
        """

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi

        if len(boxes.size()) == 2:
            boxes = boxes.unsqueeze(0)

        if normalize:
            boxes = boxes / (boxes[:, -1:] + 1e-6) * scale

        dim_t = torch.arange(embedding_dim, dtype=torch.float32, device=device)
        dim_t = temperature ** (2 * self._torch_int_div(dim_t, 2) / embedding_dim)

        pos_emb = boxes[:, :, :, None] / dim_t

        pos_emb = torch.stack(
            (pos_emb[:, :, :, 0::2].sin(), pos_emb[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos_emb = pos_emb.squeeze(0).flatten(1)

        return pos_emb

    def _learned_pos_embedding(
        self,
        boxes: torch.Tensor,
        feature_dim_attn_head: int = 1024,
        learn_pos_emb_num: int = 16,
        device: str = "cpu",
        over_boxes: bool = True,
    ) -> torch.Tensor:
        """
        Generates learned positional embeddings for boxes using given parameters.
        Args:
            boxes: the input boxes.
            feature_dim_attn_head: Number of features in attention head.
            learn_pos_emb_num: Size of the dictionary of embeddings.
            device: the device to be used (e.g., "cuda", "cpu").
            over_boxes: If True, use box dimensions, rather than box offset and shape.
        Returns:
            torch.Tensor, the learned positional embeddings.
        """

        pos_lookup = torch.nn.Embedding(
            learn_pos_emb_num * 4, feature_dim_attn_head // 4
        )

        N = boxes.shape[0]
        boxes = boxes.view(N, 4)

        if over_boxes:
            xywh = boxes
        else:
            xywh = torch.cat(
                [(boxes[:, 2:] + boxes[:, :2]) / 2, (boxes[:, 2:] - boxes[:, :2])],
                dim=1,
            )

        l, r, lw, rw = self._compute_weights(xywh, learn_pos_emb_num, device)

        f = pos_lookup.weight.shape[1]

        pos_emb_table = pos_lookup.weight.view(
            learn_pos_emb_num, 4, f
        )  # T x 4 x (D * 4)

        pos_le = pos_emb_table.gather(0, l[:, :, None].expand(N, 4, f))  # N x 4 x d
        pos_re = pos_emb_table.gather(0, r[:, :, None].expand(N, 4, f))  # N x 4 x d
        pos_emb = lw[:, :, None] * pos_re + rw[:, :, None] * pos_le

        pos_emb = pos_emb.view(N, 4 * f)

        return pos_emb

    def _learned_temp_embedding(
        self,
        times: torch.Tensor,
        feature_dim_attn_head: int = 1024,
        learn_temp_emb_num: int = 16,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generates learned temporal embeddings for times using given parameters.
        Args:
            times: the input times.
            feature_dim_attn_head: Number of features in attention head.
            learn_temp_emb_num: Size of the dictionary of embeddings.
            device: the device to be used (e.g., "cuda", "cpu").
        Returns:
            torch.Tensor, the learned temporal embeddings.
        """

        temp_lookup = torch.nn.Embedding(learn_temp_emb_num, feature_dim_attn_head)

        N = times.shape[0]

        l, r, lw, rw = self._compute_weights(times, learn_temp_emb_num, device)

        le = temp_lookup.weight[l]  # T x D --> N x D
        re = temp_lookup.weight[r]

        temp_emb = lw[:, None] * re + rw[:, None] * le

        return temp_emb.view(N, feature_dim_attn_head)

    def _compute_weights(
        self, data: torch.Tensor, learn_emb_num: int = 16, device: str = "cpu"
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generates left and right learned embedding weights.
        Args:
            data: the input data (e.g boxes or times).
            learn_temp_emb_num: Size of the dictionary of embeddings.
            device: the device to be used (e.g., "cuda", "cpu").
        Returns:
            A torch.Tensor for each of the left/right indices and weights, respectively
        """

        data = (data * learn_emb_num).to(device)

        l = data.clamp(min=0, max=learn_emb_num - 1).long()  # N x 4
        r = (l + 1).clamp(min=0, max=learn_emb_num - 1).long()  # N x 4

        lw = data - l.float()  # N x 4

        rw = 1.0 - lw

        return l, r, lw, rw
