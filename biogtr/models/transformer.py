"""
DETR Transformer class.

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

* Modified from https://github.com/facebookresearch/detr/blob/main/models/transformer.py
* Modified from https://github.com/xingyizhou/GTR/blob/master/gtr/modeling/roi_heads/transformer.py
* Modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
    * added fixed embeddings over boxes
"""


from biogtr.models.attention_head import ATTWeightHead
from biogtr.models.embedding import Embedding
from biogtr.models.model_utils import get_boxes_times
from torch import nn
from typing import Dict, List, Tuple
import copy
import torch
import torch.nn.functional as F

# todo: add named tensors
# todo: add flash attention


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        norm: bool = False,
        feature_dim_attn_head: int = 1024,
        num_layers_attn_head: int = 2,
        dropout_attn_head: float = 0.1,
        embedding_meta: dict = None,
        return_embedding: bool = False,
        decoder_self_attn: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        """
        Transformer module.
        Args:
            d_model: The number of features in the encoder/decoder inputs.
            nhead: The number of heads in the transfomer encoder/decoder.
            num_encoder_layers: The number of encoder-layers in the encoder.
            num_decoder_layers: The number of decoder-layers in the decoder.
            dim_feedforward: The dimension of the feedforward layers of the transformer.
            dropout: Dropout value applied to the output of transformer layers.
            activation: Activation function to use.
            return_intermediate_dec: Return intermediate layers from decoder.
            norm: If True, normalize output of encoder and decoder.
            feature_dim_attn_head: The number of features in the attention head.
            num_layers_attn_head: The number of layers in the attention head.
            dropout_attn_head: Dropout value for the attention_head.
            embedding_meta: Metadata for positional embeddings. See below.
            return_embedding: Whether to return the positional embeddings
            decoder_self_attn: If True, use decoder self attention.
            device: Device to use for computations.

            embedding_meta: By default this will be an empty dict and indicate
                that no positional embeddings should be used. To use positional
                embeddings, a dict should be passed with the type of embedding to
                use. Valid options are:
                    * learned_pos: only learned position embeddings
                    * learned_temp: only learned temporal embeddings
                    * learned_pos_temp: learned position and temporal embeddings
                    * fixed_pos: fixed sine position embeddings
                    * fixed_pos_temp: fixed sine position and learned temporal embeddings
                You can additionally pass kwargs to override the default
                embedding values (see embedding.py function methods for relevant
                embedding parameters). Example:

                    embedding_meta = {
                        'embedding_type': 'learned_pos_temp',
                        'kwargs': {
                            'learn_pos_emb_num': 16,
                            'learn_temp_emb_num': 16,
                            'over_boxes': False
                        }
                    }
                Note: Embedding features are handled directly in the forward
                pass for each case. Overriding the features through kwargs will
                likely throw errors due to incorrect tensor shapes.
        """

        self.device = device
        self.d_model = d_model

        self.embedding_meta = embedding_meta
        self.return_embedding = return_embedding

        if self.embedding_meta:
            key = "embedding_type"

            embedding_types = [
                "learned_pos",
                "learned_temp",
                "learned_pos_temp",
                "fixed_pos",
                "fixed_pos_temp",
            ]

            assert (
                key in self.embedding_meta
            ), f"Please provide an embedding type, valid options are {embedding_types}"

            provided_type = self.embedding_meta[key]

            assert (
                provided_type in embedding_types
            ), f"{provided_type} is invalid. Please choose a valid type from {embedding_types}"

            self.embedding = Embedding()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, norm
        )

        encoder_norm = nn.LayerNorm(d_model) if (norm) else None

        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # Transformer Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            norm,
            decoder_self_attn,
        )

        decoder_norm = nn.LayerNorm(d_model) if (norm) else None

        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec, decoder_norm
        )

        # Transformer attention head
        self.attn_head = ATTWeightHead(
            feature_dim=feature_dim_attn_head,
            num_layers=num_layers_attn_head,
            dropout=dropout_attn_head,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, instances, query_frame=None):
        """A forward pass through the transformer and attention head.
        Args:
            instances: A list of dictionaries, one dictionary for each frame
            query_frame: An integer (k) specifying the frame within the window to be queried.

        Returns:
            asso_output: A list of torch.Tensors of shape (L, N_t, N) where:
                L: number of decoder blocks
                N_t: number of instances in current query/frame
                N: number of instances in window

        # ------------------------- An example of instances ------------------------ #
        instances = [
            {
                # Each dictionary is a frame.
                "frame_id": frame index int,
                "num_detected": N_i, # num of detected instances in i-th frame
                "bboxes": (N_i, 4),  # in pascal_voc unrounded unnormalized
                "features": (N_i, D), # D = embedding dimension
                ...
            },
            ...
        ]
        """

        reid_features = torch.cat(
            [frame["features"] for frame in instances], dim=0
        ).unsqueeze(0)

        T = len(instances)
        n_t = [frame["num_detected"] for frame in instances]
        N = sum(n_t)
        D = reid_features.shape[-1]

        if self.embedding_meta:
            kwargs = self.embedding_meta.get("kwargs", {})

            pred_box, pred_time = get_boxes_times(instances, self.device)  # N x 4

            embedding_type = self.embedding_meta["embedding_type"]

            if "temp" in embedding_type:
                temp_emb = self.embedding._learned_temp_embedding(
                    pred_time / T, features=self.d_model, **kwargs
                )

                pos_emb = temp_emb

            if "learned" in embedding_type:
                if "pos" in embedding_type:
                    pos_emb = self.embedding._learned_pos_embedding(
                        pred_box, features=self.d_model, **kwargs
                    )

            else:
                pos_emb = self.embedding._sine_box_embedding(
                    pred_box, features=self.d_model // 4, **kwargs
                )

            if "temp" in embedding_type and embedding_type != "learned_temp":
                pos_emb = (pos_emb + temp_emb) / 2.0

            pos_emb = pos_emb.view(1, N, D)
            pos_emb = pos_emb.permute(1, 0, 2)  # (N, B, D)
        else:
            pos_emb = None

        query_inds = None
        N_t = N
        if query_frame is not None:
            c = query_frame
            query_inds = [x for x in range(sum(n_t[:c]), sum(n_t[: c + 1]))]
            N_t = len(query_inds)

        B, N, D = reid_features.shape
        reid_features = reid_features.permute(1, 0, 2)  # (N x B x D)

        memory = self.encoder(reid_features, pos_emb=pos_emb)  # (N, B, D)

        if query_inds is not None:
            tgt = reid_features[query_inds]
            if pos_emb is not None:
                tgt_pos_emb = pos_emb[query_inds]
            else:
                tgt_pos_emb = pos_emb
        else:
            tgt = reid_features
            tgt_pos_emb = pos_emb

        # tgt: (N_t, B, D)

        hs = self.decoder(
            tgt, memory, pos_emb=pos_emb, tgt_pos_emb=tgt_pos_emb
        )  # (L, N_t, B, D)

        feats = hs.transpose(1, 2)  # # (L, B, N_t, D)
        memory = memory.permute(1, 0, 2).view(B, N, D)  # (B, N, D)

        asso_output = []
        for x in feats:
            # x: (B=1, N_t, D=512)

            asso_output.append(self.attn_head(x, memory).view(N_t, N))

        # (L=1, N_t, N)
        return (asso_output, pos_emb) if self.return_embedding else asso_output


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None
    ):
        super().__init__()

        """
        Args:
            encoder_layer: An instance of the TransformerDecoderLayer.
            num_layers: The number of encoder layers to be stacked.
            norm: The normalization layer to be applied.
        """

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, pos_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: The input tensor of shape (N_t, B, D).
            pos_emb: The positional embedding tensor of shape (N_t, D).

        Returns:
            The output tensor of shape (N_t, B, D).
        """
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos_emb)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        return_intermediate: bool = False,
        norm: nn.Module = None,
    ):
        super().__init__()

        """
        Args:
            decoder_layer: An instance of TransformerDecoderLayer.
            num_layers: The number of decoder layers to be stacked.
            return_intermediate: Return intermediate layers from decoder.
            norm: The normalization layer to be applied.
        """

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm if norm is not None else nn.Identity()

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, pos_emb=None, tgt_pos_emb=None
    ):
        """
        Args:
            tgt: Target sequence for decoder to generate (N_t, B, D).
            memory: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (N, B, D)
            pos_emb: The input positional embedding tensor of shape (N_t, D).
            tgt_pos_emb: The target positional embedding of shape (N_t, D)

        Returns:
            The output tensor of shape (L, N_t, B, D).
        """

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                pos=pos_emb,
                tgt_pos=tgt_pos_emb,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        norm: bool = False,
    ):
        super().__init__()

        """
        Args:
            d_model: The number of features in the encoder inputs.
            nhead: The number of heads for the encoder.
            dim_feedforward: Dimension of the feedforward layers of encoder.
            dropout: Dropout value applied to the output of encoder.
            activation: Activation function to use.
            norm: If True, normalize output of encoder.
        """

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src: torch.Tensor, pos: torch.Tensor = None):
        """
        Args:
            src: Input sequence for encoder (N_t, B, D).
            pos: Position embedding, if provided is added to src

        Returns:
            The output tensor of shape (N_t, B, D).
        """

        src = src if pos is None else src + pos
        q = k = src

        src2 = self.self_attn(
            q,
            k,
            value=src,
        )[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        norm: bool = False,
        decoder_self_attn: bool = False,
    ):
        super().__init__()

        """
        Args:
            d_model: The number of features in the decoder inputs.
            nhead: The number of heads for the decoder.
            dim_feedforward: Dimension of the feedforward layers of decoder.
            dropout: Dropout value applied to the output of decoder.
            activation: Activation function to use.
            norm: If True, normalize output of decoder.
            decoder_self_attn: If True, use decoder self attention
        """

        self.decoder_self_attn = decoder_self_attn

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if self.decoder_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, pos=None, tgt_pos=None):
        """
        Args:
            tgt: Target sequence for decoder to generate (N_t, B, D).
            memory: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (N, B, D)
            pos_emb: The input positional embedding tensor of shape (N_t, D).
            tgt_pos_emb: The target positional embedding of shape (N_t, D)

        Returns:
            The output tensor of shape (N_t, B, D).
        """

        tgt = tgt if tgt_pos is None else tgt + tgt_pos
        memory = memory if pos is None else memory + pos

        q = k = tgt

        if self.decoder_self_attn:
            tgt2 = self.self_attn(q, k, value=tgt)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=tgt,  # (N_t, B, D)
            key=memory,  # (N, B, D)
            value=memory,  # (N, B, D)
        )[
            0
        ]  # (N_t, B, D)

        tgt = tgt + self.dropout2(tgt2)  # (N_t, B, D)
        tgt = self.norm2(tgt)  # (N_t, B, D)
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(tgt)))
        )  # (N_t, B, D)
        tgt = tgt + self.dropout3(tgt2)  # (N_t, B, D)
        tgt = self.norm3(tgt)

        return tgt  # (N_t, B, D)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    Args:
        module: The module to be copied.
        N: The number of copies to be made.

    Returns:
        A ModuleList containing N copies of the given module.
    """

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> callable:
    """
    Args:
        activation: The string name of the activation function to use

    Returns:
        The appropriate activation function
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
