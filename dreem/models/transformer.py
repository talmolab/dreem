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

import copy

import torch
import torch.nn.functional as F
from torch import nn

from dreem.io import AssociationMatrix, Instance
from dreem.models.attention_head import ATTWeightHead
from dreem.models.embedding import Embedding, FourierPositionalEmbeddings
from dreem.models.model_utils import get_boxes, get_times

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
        embedding_meta: dict | None = None,
        return_embedding: bool = False,
        decoder_self_attn: bool = False,
        encoder_cfg: dict | None = None,
    ) -> None:
        """Initialize Transformer.

        Args:
            d_model: The number of features in the encoder/decoder inputs.
            nhead: The number of heads in the transformer encoder/decoder.
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
            encoder_cfg: Encoder configuration.

                More details on `embedding_meta`:
                    By default this will be an empty dict and indicate
                    that no positional embeddings should be used. To use the positional embeddings
                    pass in a dictionary containing a "pos" and "temp" key with subdictionaries for correct parameters ie:
                    {"pos": {'mode': 'learned', 'emb_num': 16, 'over_boxes: 'True'},
                    "temp": {'mode': 'learned', 'emb_num': 16}}. (see `dreem.models.embeddings.Embedding.EMB_TYPES`
                    and `dreem.models.embeddings.Embedding.EMB_MODES` for embedding parameters).
        """
        super().__init__()

        self.d_model = dim_feedforward = feature_dim_attn_head = d_model

        self.embedding_meta = embedding_meta
        self.return_embedding = return_embedding
        self.encoder_cfg = encoder_cfg

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

        self.fourier_embeddings = FourierPositionalEmbeddings(
            n_components=8, d_model=d_model
        )

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, norm
        )

        encoder_norm = nn.LayerNorm(d_model) if (norm) else None

        # only used if using descriptor visual encoder; default resnet encoder uses d_model directly
        if self.encoder_cfg and "encoder_type" in self.encoder_cfg:
            self.visual_feat_dim = (
                self.encoder_cfg["ndim"] if "ndim" in self.encoder_cfg else 5
            )  # 5 is default for descriptor
            self.fourier_proj = nn.Linear(self.d_model + self.visual_feat_dim, d_model)
            self.fourier_norm = nn.LayerNorm(self.d_model)

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
            if not torch.nn.parameter.is_lazy(p) and p.dim() > 1:
                try:
                    nn.init.xavier_uniform_(p)
                except ValueError as e:
                    print(f"Failed Trying to initialize {p}")
                    raise (e)

    def forward(
        self,
        ref_instances: list[Instance],
        query_instances: list[Instance] | None = None,
    ) -> list[AssociationMatrix]:
        """Execute a forward pass through the transformer and attention head.

        Args:
            ref_instances: A list of instance objects (See `dreem.io.Instance` for more info.)
            query_instances: An set of instances to be used as decoder queries.

        Returns:
            asso_output: A list of torch.Tensors of shape (L, n_query, total_instances) where:
                L: number of decoder blocks
                n_query: number of instances in current query/frame
                total_instances: number of instances in window
        """
        ref_features = torch.cat(
            [instance.features for instance in ref_instances], dim=0
        ).unsqueeze(0)

        # window_length = len(frames)
        # instances_per_frame = [frame.num_detected for frame in frames]
        total_instances = len(ref_instances)
        embed_dim = self.d_model
        # print(f'T: {window_length}; N: {total_instances}; N_t: {instances_per_frame} n_reid: {reid_features.shape}')
        ref_boxes = get_boxes(ref_instances)  # total_instances, 4
        ref_boxes = torch.nan_to_num(ref_boxes, -1.0)
        ref_times, query_times = get_times(ref_instances, query_instances)

        # window_length = len(ref_times.unique())  # Currently unused but may be useful for debugging

        ref_temp_emb = self.temp_emb(ref_times)

        ref_pos_emb = self.pos_emb(ref_boxes)

        if self.return_embedding:
            for i, instance in enumerate(ref_instances):
                instance.add_embedding("pos", ref_pos_emb[i])
                instance.add_embedding("temp", ref_temp_emb[i])

        # ref_emb = (ref_pos_emb + ref_temp_emb) / 2.0
        ref_emb = ref_temp_emb

        ref_emb = ref_emb.view(1, total_instances, embed_dim)

        ref_emb = ref_emb.permute(1, 0, 2)  # (total_instances, batch_size, embed_dim)

        batch_size, total_instances = ref_features.shape[:-1]

        ref_features = ref_features.permute(
            1, 0, 2
        )  # (total_instances, batch_size, embed_dim)

        encoder_queries = ref_features

        # apply fourier embeddings if using fourier rope, OR if using descriptor (compact) visual encoder
        if (
            self.embedding_meta
            and "use_fourier" in self.embedding_meta
            and self.embedding_meta["use_fourier"]
        ) or (
            self.encoder_cfg
            and "encoder_type" in self.encoder_cfg
            and self.encoder_cfg["encoder_type"] == "descriptor"
        ):
            encoder_queries = apply_fourier_embeddings(
                encoder_queries,
                ref_times,
                self.d_model,
                self.fourier_embeddings,
                self.fourier_proj,
                self.fourier_norm,
            )

        encoder_features = self.encoder(
            encoder_queries, temp_emb=ref_emb, ref_boxes=ref_boxes, query_boxes=ref_boxes
        )  # (total_instances, batch_size, embed_dim)

        n_query = total_instances

        query_features = ref_features
        query_pos_emb = ref_pos_emb
        query_temp_emb = ref_temp_emb
        query_emb = ref_emb

        if query_instances is not None: # only during inference
            n_query = len(query_instances)

            query_features = torch.cat(
                [instance.features for instance in query_instances], dim=0
            ).unsqueeze(0)

            query_features = query_features.permute(
                1, 0, 2
            )  # (n_query, batch_size, embed_dim)

            query_boxes = get_boxes(query_instances)
            query_boxes = torch.nan_to_num(query_boxes, -1.0)
            query_temp_emb = self.temp_emb(query_times)

            query_pos_emb = self.pos_emb(query_boxes)

            # query_emb = (query_pos_emb + query_temp_emb) / 2.0
            query_emb = query_temp_emb
            query_emb = query_emb.view(1, n_query, embed_dim)
            query_emb = query_emb.permute(1, 0, 2)  # (n_query, batch_size, embed_dim)

        else:
            query_instances = ref_instances
            query_times = ref_times
            query_boxes = ref_boxes

        if self.return_embedding:
            for i, instance in enumerate(query_instances):
                instance.add_embedding("pos", query_pos_emb[i])
                instance.add_embedding("temp", query_temp_emb[i])

        # apply fourier embeddings if using fourier rope, OR if using descriptor (compact) visual encoder
        if (
            self.embedding_meta
            and "use_fourier" in self.embedding_meta
            and self.embedding_meta["use_fourier"]
        ) or (
            self.encoder_cfg
            and "encoder_type" in self.encoder_cfg
            and self.encoder_cfg["encoder_type"] == "descriptor"
        ):
            query_features = apply_fourier_embeddings(
                query_features,
                query_times,
                self.d_model,
                self.fourier_embeddings,
                self.fourier_proj,
                self.fourier_norm,
            )

        decoder_features = self.decoder(
            query_features,
            encoder_features,
            ref_temp_emb=ref_emb,
            query_temp_emb=query_emb,
            ref_boxes=ref_boxes,
            query_boxes=query_boxes,
        )  # (L, n_query, batch_size, embed_dim)

        decoder_features = decoder_features.transpose(
            1, 2
        )  # # (L, batch_size, n_query, embed_dim)
        encoder_features = encoder_features.permute(1, 0, 2).view(
            batch_size, total_instances, embed_dim
        )  # (batch_size, total_instances, embed_dim)

        asso_output = []
        for frame_features in decoder_features:
            asso_matrix = self.attn_head(frame_features, encoder_features).view(
                n_query, total_instances
            )
            asso_matrix = AssociationMatrix(asso_matrix, ref_instances, query_instances)

            asso_output.append(asso_matrix)

        # (L=1, n_query, total_instances)
        return asso_output

class RelPosAttention(nn.Module):
    """Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_relpos.py."""
    def __init__(
            self,
            dim: int,
            num_heads: int,
            attn_drop: float,
            proj_drop: float,
            rel_pos_cls: nn.Module,
            qkv_bias=False,
            qk_norm=False,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.rel_pos = rel_pos_cls if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.bias_weight = AttentionBiasWeight(dim, num_heads)

    def forward(self, q,k,v, ref_boxes, query_boxes):
        # assumes input is (n_query, batch_size, embed_dim)
        # use var of keys as during inference, there is only 1 query frame so var is meaningless
        global_feature_variance = torch.var(k, dim=0) # (batch_size, embed_dim)
        # print(f"Global feature variance: {global_feature_variance}")
        if torch.isnan(global_feature_variance).any():
            print("Global feature variance is nan")
        bias_weight = self.bias_weight(global_feature_variance)
            
        N_q, B, _ = q.shape
        N_k, _, _ = k.shape
        q,k,v = q.permute(1,0,2), k.permute(1,0,2), v.permute(1,0,2) # (batch_size, n_query, embed_dim)
        q = self.q_proj(q).reshape(B,N_q,self.num_heads, self.head_dim).permute(0,2,1,3) # (B, N, num_heads, head_dim)
        k = self.k_proj(k).reshape(B,N_k,self.num_heads, self.head_dim).permute(0,2,1,3) # (B, N, num_heads, head_dim)
        v = self.v_proj(v).reshape(B,N_k,self.num_heads, self.head_dim).permute(0,2,1,3) # (B, N, num_heads, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel_pos is not None:
            attn = self.rel_pos(attn, ref_boxes, query_boxes, bias_weight)
        attn = attn.softmax(dim=-1)
        # TODO: which one of attn drop and proj drop does pytorch MultiHeadAttention use? Discard the other one.
        attn = self.attn_drop(attn)
        x = attn @ v

        # x = x.transpose(1, 2).reshape(B, N, -1) # only supports single head for now
        x = x.squeeze()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EuclDistanceBias(nn.Module):
    """Relative position bias."""
    def __init__(self, n_heads: int):
        super().__init__()
        assert n_heads == 1, "Euclidean distance bias in the attention module is only supported for 1 attention head"

    def get_bias(self, ref_boxes, query_boxes) -> torch.Tensor:
        ref_boxes = ref_boxes.squeeze()
        query_boxes = query_boxes.squeeze()
        if query_boxes.dim() == 1:
            query_boxes = query_boxes.unsqueeze(0)
        query_centroids = (query_boxes[:,:2] + query_boxes[:,2:]) / 2
        ref_centroids = (ref_boxes[:,:2] + ref_boxes[:,2:]) / 2
        # (n_query, n_ref)
        eucl_dist = torch.cdist(query_centroids, ref_centroids, p=2)
        # need (B, n_head, n_query, n_ref)
        return eucl_dist.unsqueeze(0).unsqueeze(0)

    def forward(self, attn, ref_boxes, query_boxes, bias_weight: torch.Tensor):
        bias = self.get_bias(ref_boxes, query_boxes)
        return attn + bias * bias_weight
        

class AttentionBiasWeight(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model, n_heads),
            nn.Sigmoid()
        )
    
    def forward(self, global_feature_variance: torch.Tensor):
        return self.mlp(global_feature_variance)


def apply_fourier_embeddings(
    queries: torch.Tensor,
    times: torch.Tensor,
    d_model: int,
    fourier_embeddings: FourierPositionalEmbeddings,
    proj: nn.Linear,
    norm: nn.LayerNorm,
) -> torch.Tensor:
    """Apply fourier embeddings to queries.

    Args:
        queries: The input tensor of shape (n_query, batch_size, embed_dim).
        times: The times index tensor of shape (n_query,).
        d_model: Model dimension.
        fourier_embeddings: The Fourier positional embeddings object.
        proj: Linear projection layer that projects concatenated feature vector to model dimension.
        norm: The normalization layer.

    Returns:
        The output queries of shape (n_query, batch_size, embed_dim).
    """
    embs = fourier_embeddings(times).permute(1, 0, 2)
    cat_queries = torch.cat([queries, embs], dim=-1)
    # project to d_model
    cat_queries = proj(cat_queries)
    cat_queries = norm(cat_queries)

    return cat_queries


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
    ) -> None:
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
        self.attn_bias = EuclDistanceBias(nhead)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.biased_self_attn = RelPosAttention(d_model, nhead, dropout, dropout, self.attn_bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self, queries: torch.Tensor, temp_emb: torch.Tensor = None, ref_boxes: torch.Tensor = None, query_boxes: torch.Tensor = None
    ) -> torch.Tensor:
        """Execute a forward pass of the encoder layer.

        Args:
            queries: Input sequence for encoder (n_query, batch_size, embed_dim).
            temp_emb: Temporal embedding, if provided is added to queries
            ref_boxes: The input boxes tensor of shape (total_instances, batch_size, 4).
            query_boxes: The target boxes tensor of shape (n_query, batch_size, 4).
        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """
        if temp_emb is None:
            temp_emb = torch.zeros_like(queries)

        queries = queries + temp_emb # these are now just temporal embeddings

        attn_features = self.biased_self_attn(
            q=queries,
            k=queries,
            v=queries,
            ref_boxes=ref_boxes,
            query_boxes=query_boxes
        )[0]

        queries = queries + self.dropout1(attn_features)
        queries = self.norm1(queries)
        projection = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        queries = queries + self.dropout2(projection)
        encoder_features = self.norm2(queries)

        return encoder_features


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
    ) -> None:
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

        self.attn_bias = EuclDistanceBias(nhead)
        self.biased_cross_attn = RelPosAttention(d_model, nhead, dropout, dropout, self.attn_bias)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if self.decoder_self_attn: # don't add bias to decoder self attn since only query instances in current frame are considered
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        decoder_queries: torch.Tensor,
        encoder_features: torch.Tensor,
        ref_temp_emb: torch.Tensor | None = None,
        query_temp_emb: torch.Tensor | None = None,
        ref_boxes: torch.Tensor = None,
        query_boxes: torch.Tensor = None,
    ) -> torch.Tensor:
        """Execute forward pass of decoder layer.

        Args:
            decoder_queries: Target sequence for decoder to generate (n_query, batch_size, embed_dim).
            encoder_features: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (total_instances, batch_size, embed_dim)
            ref_temp_emb: The input temporal embedding tensor of shape (n_query, embed_dim).
            query_temp_emb: The target temporal embedding of shape (n_query, embed_dim)
            ref_boxes: The input boxes tensor of shape (total_instances, batch_size, 4).
            query_boxes: The target boxes tensor of shape (n_query, batch_size, 4).

        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """
        if query_temp_emb is None:
            query_temp_emb = torch.zeros_like(decoder_queries)
        if ref_temp_emb is None:
            ref_temp_emb = torch.zeros_like(encoder_features)

        decoder_queries = decoder_queries + query_temp_emb # note these are only temporal embeddings now
        encoder_features = encoder_features + ref_temp_emb

        if self.decoder_self_attn:
            self_attn_features = self.self_attn(
                query=decoder_queries, key=decoder_queries, value=decoder_queries
            )[0]
            decoder_queries = decoder_queries + self.dropout1(self_attn_features)
            decoder_queries = self.norm1(decoder_queries)

        # x_attn_features = self.biased_cross_attn(
        #     query=decoder_queries,  # (n_query, batch_size, embed_dim)
        #     key=encoder_features,  # (total_instances, batch_size, embed_dim)
        #     value=encoder_features,  # (total_instances, batch_size, embed_dim)
        # )[0]  # (n_query, batch_size, embed_dim)

        x_attn_features = self.biased_cross_attn(
            q=decoder_queries,  # (n_query, batch_size, embed_dim)
            k=encoder_features,  # (total_instances, batch_size, embed_dim)
            v=encoder_features,  # (total_instances, batch_size, embed_dim)
            ref_boxes=ref_boxes,
            query_boxes=query_boxes,
        )[0]  # (n_query, batch_size, embed_dim)

        decoder_queries = decoder_queries + self.dropout2(
            x_attn_features
        )  # (n_query, batch_size, embed_dim)
        decoder_queries = self.norm2(
            decoder_queries
        )  # (n_query, batch_size, embed_dim)
        projection = self.linear2(
            self.dropout(self.activation(self.linear1(decoder_queries)))
        )  # (n_query, batch_size, embed_dim)
        decoder_queries = decoder_queries + self.dropout3(
            projection
        )  # (n_query, batch_size, embed_dim)
        decoder_features = self.norm3(decoder_queries)

        return decoder_features  # (n_query, batch_size, embed_dim)


class TransformerEncoder(nn.Module):
    """A transformer encoder block composed of encoder layers."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        """Initialize transformer encoder.

        Args:
            encoder_layer: An instance of the TransformerEncoderLayer.
            num_layers: The number of encoder layers to be stacked.
            norm: The normalization layer to be applied.
        """
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm if norm is not None else nn.Identity()

    def forward(
        self, queries: torch.Tensor, temp_emb: torch.Tensor = None, ref_boxes: torch.Tensor = None, query_boxes: torch.Tensor = None
    ) -> torch.Tensor:
        """Execute a forward pass of encoder layer.

        Args:
            queries: The input tensor of shape (n_query, batch_size, embed_dim).
            temp_emb: The temporal embedding tensor of shape (n_query, embed_dim).
            ref_boxes: The input boxes tensor of shape (total_instances, batch_size, 4).
            query_boxes: The target boxes tensor of shape (n_query, batch_size, 4).
        Returns:
            The output tensor of shape (n_query, batch_size, embed_dim).
        """
        for layer in self.layers:
            queries = layer(queries, temp_emb=temp_emb, ref_boxes=ref_boxes, query_boxes=query_boxes)

        encoder_features = self.norm(queries)
        return encoder_features


class TransformerDecoder(nn.Module):
    """Transformer Decoder Block composed of Transformer Decoder Layers."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
        norm: nn.Module | None = None,
    ) -> None:
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
        self,
        decoder_queries: torch.Tensor,
        encoder_features: torch.Tensor,
        ref_temp_emb: torch.Tensor | None = None,
        query_temp_emb: torch.Tensor | None = None,
        ref_boxes: torch.Tensor = None,
        query_boxes: torch.Tensor = None,
    ) -> torch.Tensor:
        """Execute a forward pass of the decoder block.

        Args:
            decoder_queries: Query sequence for decoder to generate (n_query, batch_size, embed_dim).
            encoder_features: Output from encoder, that decoder uses to attend to relevant
                parts of input sequence (total_instances, batch_size, embed_dim)
            ref_temp_emb: The input temporal embedding tensor of shape (total_instances, batch_size, embed_dim).
            query_temp_emb: The query temporal embedding of shape (n_query, batch_size, embed_dim)
            ref_boxes: The input boxes tensor of shape (total_instances, batch_size, 4).
            query_boxes: The target boxes tensor of shape (n_query, batch_size, 4).

        Returns:
            The output tensor of shape (L, n_query, batch_size, embed_dim).
        """
        decoder_features = decoder_queries

        intermediate = []

        for layer in self.layers:
            decoder_features = layer(
                decoder_features,
                encoder_features,
                ref_temp_emb=ref_temp_emb,
                query_temp_emb=query_temp_emb,
                ref_boxes=ref_boxes,
                query_boxes=query_boxes,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(decoder_features))

        decoder_features = self.norm(decoder_features)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(decoder_features)

            return torch.stack(intermediate)

        return decoder_features.unsqueeze(0)


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
