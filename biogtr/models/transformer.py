"""DETR Transformer class.

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

* Modified from https://github.com/facebookresearch/detr/blob/main/models/transformer.py
* Modified from https://github.com/xingyizhou/GTR/blob/master/gtr/modeling/roi_heads/transformer.py
* Modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
    * added fixed embeddings over boxes
"""

from biogtr.data_structures import Frame
from biogtr.models.attention_head import ATTWeightHead
from biogtr.models.embedding import Embedding
from biogtr.models.model_utils import get_boxes_times
from torch import nn
import copy
import torch
import torch.nn.functional as F

# todo: add named tensors
# todo: add flash attention


class Transformer(torch.nn.Module):
    """Transformer class."""

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        norm: bool = False,
        num_layers_attn_head: int = 2,
        dropout_attn_head: float = 0.1,
        embedding_meta: dict = None,
        return_embedding: bool = False,
        decoder_self_attn: bool = False,
    ):
        """Initialize Transformer.

        Args:
            d_model: The number of features in the encoder/decoder inputs.
            nhead: The number of heads in the transfomer encoder/decoder.
            num_encoder_layers: The number of encoder-layers in the encoder.
            num_decoder_layers: The number of decoder-layers in the decoder.
            dropout: Dropout value applied to the output of transformer layers.
            activation: Activation function to use.
            return_intermediate_dec: Return intermediate layers from decoder.
            norm: If True, normalize output of encoder and decoder.
            num_layers_attn_head: The number of layers in the attention head.
            dropout_attn_head: Dropout value for the attention_head.
            embedding_meta: Metadata for positional embeddings. See below.
            return_embedding: Whether to return the positional embeddings
            decoder_self_attn: If True, use decoder self attention.

            More details on `embedding_meta`:
                By default this will be an empty dict and indicate
                that no positional embeddings should be used. To use the positional embeddings
                pass in a dictionary containing a "pos" and "temp" key with subdictionaries for correct parameters ie:
                {"pos": {'mode': 'learned', 'emb_num': 16, 'over_boxes: 'True'},
                "temp": {'mode': 'learned', 'emb_num': 16}}. (see `biogtr.models.embeddings.Embedding.EMB_TYPES`
                and `biogtr.models.embeddings.Embedding.EMB_MODES` for embedding parameters).
        """
        super().__init__()

        self.d_model = dim_feedforward = feature_dim_attn_head = d_model

        self.embedding_meta = embedding_meta
        self.return_embedding = return_embedding

        self.pos_emb = Embedding(emb_type="off", mode="off", features=self.d_model)
        self.temp_emb = Embedding(emb_type="off", mode="off", features=self.d_model)

        if self.embedding_meta:
            if "pos" in self.embedding_meta:
                pos_emb_cfg = self.embedding_meta["pos"]
                if pos_emb_cfg:
                    self.pos_emb = Embedding(
                        emb_type="pos", features=self.d_model, **pos_emb_cfg
                    )
            if "temp" in self.embedding_meta:
                temp_emb_cfg = self.embedding_meta["temp"]
                if temp_emb_cfg:
                    self.temp_emb = Embedding(
                        emb_type="temp", features=self.d_model, **temp_emb_cfg
                    )

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
        """Initialize model weights from xavier distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, frames: list[Frame], query_frame: int = None):
        """Execute a forward pass through the transformer and attention head.

        Args:
            frames: A list of Frames (See `biogtr.data_structures.Frame for more info.)
            query_frame: An integer (k) specifying the frame within the window to be queried.

        Returns:
            asso_output: A list of torch.Tensors of shape (L, n_query, total_instances) where:
                L: number of decoder blocks
                n_query: number of instances in current query/frame
                total_instances: number of instances in window
        """
        try:
            reid_features = torch.cat(
                [frame.get_features() for frame in frames], dim=0
            ).unsqueeze(0)
        except Exception as e:
            print([[f.device for f in frame.get_features()] for frame in frames])
            raise (e)

        window_length = len(frames)
        instances_per_frame = [frame.num_detected for frame in frames]
        total_instances = sum(instances_per_frame)
        embed_dim = reid_features.shape[-1]

        # print(f'T: {window_length}; N: {total_instances}; N_t: {instances_per_frame} n_reid: {reid_features.shape}')
        pred_box, pred_time = get_boxes_times(frames)  # total_instances x 4

        temp_emb = self.temp_emb(pred_time / window_length)

        pos_emb = self.pos_emb(pred_box)

        try:
            emb = (pos_emb + temp_emb) / 2.0
        except RuntimeError as e:
            print(self.pos_emb.features, self.temp_emb.features)
            print(pos_emb.shape, temp_emb.shape)
            raise (e)

        emb = emb.view(1, total_instances, embed_dim)

        emb = emb.permute(1, 0, 2)  # (total_instances, batch_size, embed_dim)

        query_inds = None
        n_query = total_instances
        if query_frame is not None:
            query_inds = [
                x
                for x in range(
                    sum(instances_per_frame[:query_frame]),
                    sum(instances_per_frame[: query_frame + 1]),
                )
            ]
            n_query = len(query_inds)

        batch_size, total_instances, embed_dim = reid_features.shape
        reid_features = reid_features.permute(
            1, 0, 2
        )  # (total_instances x batch_size x embed_dim)

        memory = self.encoder(
            reid_features, pos_emb=emb
        )  # (total_instances, batch_size, embed_dim)

        tgt = reid_features
        tgt_emb = emb

        if query_inds is not None:
            tgt = tgt[query_inds]  # tgt: (n_query, batch_size, embed_dim)
            tgt_emb = tgt_emb[query_inds]

        hs = self.decoder(
            tgt, memory, pos_emb=emb, tgt_pos_emb=tgt_emb
        )  # (L, n_query, batch_size, embed_dim)

        feats = hs.transpose(1, 2)  # # (L, batch_size, n_query, embed_dim)
        memory = memory.permute(1, 0, 2).view(
            batch_size, total_instances, embed_dim
        )  # (batch_size, total_instances, embed_dim)

        asso_output = []
        for x in feats:
            # x: (batch_size=1, n_query, embed_dim=512)

            asso_output.append(self.attn_head(x, memory).view(n_query, total_instances))

        # (L=1, n_query, total_instances)
        return (asso_output, emb) if self.return_embedding else (asso_output, None)


class TransformerEncoder(nn.Module):
    """A transformer encoder block composed of encoder layers."""

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module = None
    ):
        """Initialize transformer encoder.

        Args:
            encoder_layer: An instance of the TransformerEncoderLayer.
            num_layers: The number of encoder layers to be stacked.
            norm: The normalization layer to be applied.
        """
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, pos_emb: torch.Tensor = None) -> torch.Tensor:
        """Execute a forward pass of encoder layer.

        Args:
            src: The input tensor of shape (n_query, batch_size, embed_dim).
            pos_emb: The positional embedding tensor of shape (n_query, embed_dim).

        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos_emb)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Transformer Decoder Block composed of Transformer Decoder Layers."""

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        return_intermediate: bool = False,
        norm: nn.Module = None,
    ):
        """Initialize transformer decoder block.

        Args:
            decoder_layer: An instance of TransformerDecoderLayer.
            num_layers: The number of decoder layers to be stacked.
            return_intermediate: Return intermediate layers from decoder.
            norm: The normalization layer to be applied.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm if norm is not None else nn.Identity()

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor, pos_emb=None, tgt_pos_emb=None
    ):
        """Execute a forward pass of the decoder block.

        Args:
            tgt: Target sequence for decoder to generate (n_query, batch_size, embed_dim).
            memory: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (total_instances, batch_size, embed_dim)
            pos_emb: The input positional embedding tensor of shape (n_query, embed_dim).
            tgt_pos_emb: The target positional embedding of shape (n_query, embed_dim)

        Returns:
            The output tensor of shape (L, n_query, batch_size, embed_dim).
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
    """A single transformer encoder layer."""

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        norm: bool = False,
    ):
        """Initialize a transformer encoder layer.

        Args:
            d_model: The number of features in the encoder inputs.
            nhead: The number of heads for the encoder.
            dim_feedforward: Dimension of the feedforward layers of encoder.
            dropout: Dropout value applied to the output of encoder.
            activation: Activation function to use.
            norm: If True, normalize output of encoder.
        """
        super().__init__()
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
        """Execute a forward pass of the encoder layer.

        Args:
            src: Input sequence for encoder (n_query, batch_size, embed_dim).
            pos: Position embedding, if provided is added to src

        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
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
    """A single transformer decoder layer."""

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
        """Initialize transformer decoder layer.

        Args:
            d_model: The number of features in the decoder inputs.
            nhead: The number of heads for the decoder.
            dim_feedforward: Dimension of the feedforward layers of decoder.
            dropout: Dropout value applied to the output of decoder.
            activation: Activation function to use.
            norm: If True, normalize output of decoder.
            decoder_self_attn: If True, use decoder self attention
        """
        super().__init__()

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
        """Execute forward pass of decoder layer.

        Args:
            tgt: Target sequence for decoder to generate (n_query, batch_size, embed_dim).
            memory: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (total_instances, batch_size, embed_dim)
            pos_emb: The input positional embedding tensor of shape (n_query, embed_dim).
            tgt_pos_emb: The target positional embedding of shape (n_query, embed_dim)

        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """
        tgt = tgt if tgt_pos is None else tgt + tgt_pos
        memory = memory if pos is None else memory + pos

        q = k = tgt

        if self.decoder_self_attn:
            tgt2 = self.self_attn(q, k, value=tgt)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=tgt,  # (n_query, batch_size, embed_dim)
            key=memory,  # (total_instances, batch_size, embed_dim)
            value=memory,  # (total_instances, batch_size, embed_dim)
        )[
            0
        ]  # (n_query, batch_size, embed_dim)

        tgt = tgt + self.dropout2(tgt2)  # (n_query, batch_size, embed_dim)
        tgt = self.norm2(tgt)  # (n_query, batch_size, embed_dim)
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(tgt)))
        )  # (n_query, batch_size, embed_dim)
        tgt = tgt + self.dropout3(tgt2)  # (n_query, batch_size, embed_dim)
        tgt = self.norm3(tgt)

        return tgt  # (n_query, batch_size, embed_dim)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Generate repeated clones of same layer type.

    Args:
        module: The module to be copied.
        N: The number of copies to be made.

    Returns:
        A ModuleList containing N copies of the given module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> callable:
    """Choose activation function to be used.

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
