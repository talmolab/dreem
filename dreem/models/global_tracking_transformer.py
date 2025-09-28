"""Module containing GTR model used for training."""

from typing import TYPE_CHECKING

import torch

from dreem.models.transformer import Transformer
from dreem.models.visual_encoder import create_visual_encoder

if TYPE_CHECKING:
    from dreem.io import AssociationMatrix, Instance, Frame

# todo: do we want to handle params with configs already here?


class GlobalTrackingTransformer(torch.nn.Module):
    """Modular GTR model composed of visual encoder + transformer used for tracking."""

    def __init__(
        self,
        encoder_cfg: dict | None = None,
        d_model: int = 1024,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: int = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = False,
        norm: bool = False,
        num_layers_attn_head: int = 2,
        dropout_attn_head: int = 0.1,
        embedding_meta: dict | None = None,
        return_embedding: bool = False,
        decoder_self_attn: bool = False,
        crop_size: int = None,
    ):
        """Initialize GTR.

        Args:
            encoder_cfg: Dictionary of arguments to pass to the CNN constructor,
                e.g: `cfg = {"model_name": "resnet18", "pretrained": False, "in_chans": 3}`
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
            crop_size: The size of the crops to be used for the visual encoder.
                More details on `embedding_meta`:
                    By default this will be an empty dict and indicate
                    that no positional embeddings should be used. To use the positional embeddings
                    pass in a dictionary containing a "pos" and "temp" key with subdictionaries for correct parameters ie:
                    `{"pos": {'mode': 'learned', 'emb_num': 16, 'over_boxes: True},
                    "temp": {'mode': 'learned', 'emb_num': 16}}`. (see `dreem.models.embeddings.Embedding.EMB_TYPES`
                    and `dreem.models.embeddings.Embedding.EMB_MODES` for embedding parameters).
        """
        super().__init__()

        if not encoder_cfg:
            encoder_cfg = {}
        self.visual_encoder = create_visual_encoder(d_model=d_model, crop_size=crop_size, **encoder_cfg)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            norm=norm,
            num_layers_attn_head=num_layers_attn_head,
            dropout_attn_head=dropout_attn_head,
            embedding_meta=embedding_meta,
            return_embedding=return_embedding,
            decoder_self_attn=decoder_self_attn,
            encoder_cfg=encoder_cfg,
        )

    def forward(
        self, frames: list["Frame"], ref_instances: list["Instance"], query_instances: list["Instance"] = None
    ) -> list["AssociationMatrix"]:
        """Execute forward pass of GTR Model to get asso matrix.

        Args:
            frames: List of frames containing instances and other data needed for transformer model
            ref_instances: List of instances from chunk containing crops of objects + gt label info
            query_instances: list of instances used as query in decoder.

        Returns:
            An N_T x N association matrix
        """
        # Extract feature representations with pre-trained encoder.
        self.extract_features(frames, ref_instances)

        if query_instances:
            self.extract_features(frames, query_instances, is_query=True)

        asso_preds = self.transformer(ref_instances, query_instances)

        return asso_preds

    def extract_features(
        self, frames: list["Frame"], instances: list["Instance"], force_recompute: bool = False, is_query: bool = False
    ) -> None:
        """Extract features from instances using visual encoder backbone.

        Args:
            frames: List of frames containing instances and other data needed for transformer model
            instances: A list of instances to compute features for. During training, this is all the instances from the context window.
            During inference, this is only the query instances i.e. from the current frame.
            force_recompute: indicate whether to compute features for all instances regardless of if they have instances
            is_query: indicate whether the instances are from the query frame
        """
        if len(instances) == 0:
            return

        bboxes = []
        images = []
        for frame in frames: # for frame in tracked_frames
            if is_query:
                frame = frames[-1]
            frame_bboxes = []
            for instance in frame.instances: # for all instances in this past tracked frame (ONLY CONTAINS INSTANCES
                # THAT NEEDED TO BE INCLUDED IN CONTEXT WINDOW)
                raw_bbox = instance.bbox.squeeze()
                # torch expects x1,y1,x2,y2 but instance.bbox is y1,x1,y2,x2
                bbox = torch.tensor([[raw_bbox[1], raw_bbox[0], raw_bbox[3], raw_bbox[2]]], device=instance.device)
                frame_bboxes.append(bbox)
            images.append(frame.img.to(instance.device))
            bboxes.append(torch.concatenate(frame_bboxes, dim=0))
            # bboxes is list of Tensor[num_instances, 4]
            if is_query:
                break

        images = torch.stack(images, dim=0) # (B, C, H, W)

        features = self.visual_encoder(images, bboxes)
        features = features.to(device=instances[0].device)

        for i, instance in enumerate(instances):
            instance.features = features[i]
