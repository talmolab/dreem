"""Module containing Wrapper around all other feature extractors."""
from biogtr.models.feature_encoders.visual_encoder import VisualEncoder
from biogtr.models.feature_encoders.flow_encoder import FlowEncoder
from biogtr.models.feature_encoders.lsd_encoder import LSDEncoder
from biogtr.models.feature_encoders import fusion_layers
import torch


class FeatureEncoder(torch.nn.Module):
    """Wrapper around all other feature extractors. Handles both getting and combining features for input into transformer."""

    def __init__(
        self,
        visual_encoder_cfg: dict = {
            "model_name": "resnet50",
            "cfg": {"weights": "ResNet50_Weights.DEFAULT"},
        },
        flow_encoder_cfg: dict = None,
        lsd_encoder_cfg: dict = None,
        out_dim: int = 512,
        fusion_layer: str = "sum",
        normalize="pre",
    ):
        """Initialize Encoder.

        Args:
            visual_encoder_cfg: dictionary containing hyperparameters for constructing visual encoder. If None then visual encoder will be turned off.
            flow_encoder_cfg: dictionary containing hyperparameters for constructing optical flow encoder. If None then optical flow encoder will be turned off.
            lsd_encoder_cfg: dictionary containing hyperparameters for constructing lsd encoder. If None then lsd encoder will be turned off.
            out_dim: out dimension of feature vectors. Should match dimensions of the transformer
            fusion_layer: what type of fusion layer to use
            normalize: when to normalize feature vectors one of {"pre", "post", "", or "both"}.
        """
        super().__init__()

        normalize_post_fusion = True
        normalize_pre_fusion = True
        if "pre" in normalize:
            normalize_post_fusion = False
        elif "post" in normalize:
            normalize_pre_fusion = False
        elif normalize == "":
            normalize_post_fusion = False
            normalize_pre_fusion = False

        self.out_dim = out_dim

        if visual_encoder_cfg is not None:
            self.visual_encoder = VisualEncoder(
                d_model=out_dim, normalize=normalize_pre_fusion, **visual_encoder_cfg
            )
        else:
            self.visual_encoder = None

        if flow_encoder_cfg is not None:
            self.flow_encoder = FlowEncoder(
                d_model=out_dim, normalize=normalize_pre_fusion, cfg=flow_encoder_cfg
            )
        else:
            self.flow_encoder = None

        self.pred_lsds = False

        if lsd_encoder_cfg is not None:
            self.lsd_encoder = LSDEncoder(
                d_model=out_dim, normalize=normalize_pre_fusion, **lsd_encoder_cfg
            )
            if lsd_encoder_cfg["unet_cfg"] is None:
                self.pred_lsds = False
            else:
                self.pred_lsds = True
        else:
            self.lsd_encoder = None

        if "cat" in fusion_layer.lower():
            self.out_layer = fusion_layers.Cat(out_dim, normalize=normalize_post_fusion)
        else:
            self.out_layer = fusion_layers.Sum(normalize=normalize_post_fusion)

    def forward(
        self,
        crops: torch.Tensor = None,
        flows: torch.Tensor = None,
        lsds: torch.Tensor = None,
        feats_to_return: set[str] = set(),
    ) -> dict[torch.TensorType]:
        """Extract and combine features.

        Args:
            crops: crops around instances of shape (B, C, H, W)
            flows: optical flow of instance. Shape = (B, 2, H, W)
            lsds: Either precomputed LSDs of Shape (B, 6, H, W) or crops around instances of shape (B, C, H, W)
            feats_to_return: a set of features to return. Useful for debugging. Must be a subset of {"visual", "flow", "lsd"}

        Returns:
            A dictionary containing each feature that was computed.
            Will always contain "combined" key and optionally subset of {"visual", "flow", "lsd"} depending on `feats_to_return`
        """
        out_feats = {}

        if self.visual_encoder is not None and crops is not None and len(crops):
            try:
                vis_feats = self.visual_encoder(crops)
            except Exception as e:
                print(crops)
                raise(e)
            out_feats["visual"] = vis_feats

        if self.flow_encoder is not None and flows is not None and len(flows):
            flow_feats = self.flow_encoder(flows)

            out_feats["flow"] = flow_feats

        if self.lsd_encoder is not None and lsds is not None and len(lsds):
            lsd_feats = self.lsd_encoder(lsds)
            out_feats["lsd"] = lsd_feats

        if len(out_feats) == 0:
            out_feats["combined"] = torch.zeros((crops.shape[0], self.out_dim), device=next(self.parameters()).device)
        else:
            out_feats["combined"] = self.out_layer(list(out_feats.values()))
        return {
            feat: tensor
            for feat, tensor in out_feats.items()
            if feat == "combined" or feat in feats_to_return
        }
