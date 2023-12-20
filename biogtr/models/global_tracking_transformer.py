"""Module containing GTR model used for training."""
from biogtr.models.transformer import Transformer
from biogtr.models.feature_encoders.feature_encoder import FeatureEncoder
from biogtr.data_structures import Frame
from torch import nn

# todo: do we want to handle params with configs already here?


class GlobalTrackingTransformer(nn.Module):
    """Modular GTR model composed of visual encoder + transformer used for tracking."""

    def __init__(
        self,
        feature_encoder_cfg: dict = {
            "visual_encoder_cfg": {},
            "lsd_encoder_cfg": None,
            "flow_encoder_cfg": None,
        },
        d_model: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: int = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        feature_dim_attn_head: int = 1024,
        norm: bool = False,
        num_layers_attn_head: int = 2,
        dropout_attn_head: int = 0.1,
        embedding_meta: dict = {"pos": None, "temp": None, "rel": None},
        return_embedding: bool = False,
        decoder_self_attn: bool = False,
        **kwargs,
    ):
        """Initialize GTR.

        Args:
            feature_encoder_cfg: Dictionary of arguments to pass to the FeatureEncoder constructor,
                e.g: `{"visual_encoder_cfg": {}, "lsd_encoder_cfg": None, "flow_encoder_cfg": None}`
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
            embedding_meta: Dict containing hyperparameters for pos, temp and rel embeddings.
                            Must have {"pos", "rel", or "temp"} in keys. `embedding_meta[type]= None` represents turning off that embedding
                            Ex: {"pos": {"emb_type":"learned"}, "temp": {"emb_type":"learned"}, rel: None}
        """
        super().__init__()

        self.feature_encoder = FeatureEncoder(out_dim=d_model, **feature_encoder_cfg)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            feature_dim_attn_head=feature_dim_attn_head,
            norm=norm,
            num_layers_attn_head=num_layers_attn_head,
            dropout_attn_head=dropout_attn_head,
            embedding_meta=embedding_meta,
            return_embedding=return_embedding,
            decoder_self_attn=decoder_self_attn,
        )

    def forward(self, frames: list[Frame], query_frame: int = None):
        """Execute forward pass of GTR Model to get asso matrix.

        Args:
            frames: List of Frames from chunk containing crops of objects + gt label info
            query_frame: Frame index used as query for self attention. Only used in sliding inference where query frame is the last frame in the window.

        Returns:
            An N_T x N association matrix
        """
        # Extract feature representations with pre-trained encoder.
        for frame in frames:
            if frame.has_instances():
                if not frame.has_features():
                    crops = frame.get_crops()
                    flows = frame.get_flows()
                    if self.feature_encoder.pred_lsds:
                        lsds = crops
                    else:
                        lsds = frame.get_lsds()
                    z = self.feature_encoder(crops=crops, flows=flows, lsds=lsds)[
                        "combined"
                    ]

                    for i, z_i in enumerate(z):
                        frame.instances[i].features = z_i
            else:
                for i, instance in enumerate(frame.instances):
                    instance.features = torch.empty(0, self.feature_encoder.d_model, device=next(self.parameters()).device)

        asso_preds, emb = self.transformer(frames, query_frame=query_frame)

        return asso_preds, emb
