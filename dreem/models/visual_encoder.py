"""Module for different visual feature extractors."""

from typing import Any, Dict, Type

import numpy as np
import skimage.measure as measure
import timm
import torch
import torch.nn.functional as F
import torchvision

# import timm

# todo: do we want to make timm a dependency?
# todo: add named tensor support

ENCODER_REGISTRY: Dict[str, Type[torch.nn.Module]] = {}


class VisualEncoderROIAlign(torch.nn.Module):
    """Class wrapping around a visual feature extractor backbone.

    Currently CNN only.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        d_model: int = 512,
        in_chans: int = 3,
        backend: int = "timm",
        crop_size: int = None,
        **kwargs: Any | None,
    ):
        """Initialize Visual Encoder.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            d_model (int): Output embedding dimension.
            in_chans: the number of input channels of the image.
            backend: Which model backend to use. One of {"timm", "torchvision"}
            kwargs: see `timm.create_model` and `torchvision.models.resnetX` for kwargs.
        """
        super().__init__()

        self.model_name = model_name.lower()
        self.d_model = d_model
        self.backend = backend
        if crop_size is None:
            raise ValueError("crop_size must be specified for ROI Align")
        self.crop_size = crop_size
        if in_chans == 1:
            self.in_chans = 3
        else:
            self.in_chans = in_chans

        self.feature_extractor = self.select_feature_extractor(
            model_name=self.model_name,
            in_chans=self.in_chans,
            backend=self.backend,
            **kwargs,
        )
        self.layer_activation = {}
        self.feature_extractor.layer4.register_forward_hook(self.get_activation('layer4'))

        self.roi_align_output_size = int(self.crop_size / 32) # TODO: remove hardcoded value - do this based on downsampling rate of the layer in use
        num_feat_map_channels = self.feature_extractor.layer4[-1].conv2.out_channels # TODO: hardcoded for layer4. change this
        self.post_align_conv1 = torch.nn.Conv2d(
            in_channels=num_feat_map_channels,
            out_channels=d_model,
            kernel_size=(self.roi_align_output_size, self.roi_align_output_size),
            stride=1,
            padding=0,
        )
        self.bnorm1 = torch.nn.BatchNorm2d(d_model)
        # 1x1 conv mlp
        self.post_align_conv2 = torch.nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )
        self.bnorm2 = torch.nn.BatchNorm2d(d_model)

    def select_feature_extractor(
        self, model_name: str, in_chans: int, backend: str, **kwargs: Any
    ) -> torch.nn.Module:
        """Select the appropriate feature extractor based on config.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            in_chans: the number of input channels of the image.
            backend: Which model backend to use. One of {"timm", "torchvision"}
            kwargs: see `timm.create_model` and `torchvision.models.resnetX` for kwargs.

        Returns:
            a CNN encoder based on the config and backend selected.
        """
        if "timm" in backend.lower():
            feature_extractor = timm.create_model(
                model_name=self.model_name,
                in_chans=self.in_chans,
                num_classes=0,
                **kwargs,
            )
        elif "torch" in backend.lower():
            if model_name.lower() == "resnet18":
                feature_extractor = torchvision.models.resnet18(**kwargs)

            elif model_name.lower() == "resnet50":
                feature_extractor = torchvision.models.resnet50(**kwargs)

            else:
                raise ValueError(
                    f"Only `[resnet18, resnet50]` are available when backend is {backend}. Found {model_name}"
                )
            feature_extractor = torch.nn.Sequential(
                *list(feature_extractor.children())[:-1]
            )
            input_layer = feature_extractor[0]
            if in_chans != 3:
                feature_extractor[0] = torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=input_layer.out_channels,
                    kernel_size=input_layer.kernel_size,
                    stride=input_layer.stride,
                    padding=input_layer.padding,
                    dilation=input_layer.dilation,
                    groups=input_layer.groups,
                    bias=input_layer.bias,
                    padding_mode=input_layer.padding_mode,
                )

        else:
            raise ValueError(
                f"Only ['timm', 'torch'] backends are available! Found {backend}."
            )
        return feature_extractor

    def encoder_dim(self, model: torch.nn.Module) -> int:
        """Compute dummy forward pass of encoder model and get embedding dimension.

        Args:
            model: a vision encoder model.

        Returns:
            The embedding dimension size.
        """
        _ = model.eval()
        dummy_output = model(torch.randn(1, self.in_chans, 224, 224)).squeeze()
        _ = model.train()  # to be safe
        return dummy_output.shape[-1]

    def roi_align(self, feature_maps: torch.Tensor, bboxes: torch.Tensor, imgs_shape: torch.Tensor) -> torch.Tensor:
        """Roi align the feature map.

        Args:
            feature_maps: The feature maps to align of shape (B, C, H_F, W_F).
            bboxes: Input bounding box list of Tensor[num_instances, 4] for each frame.
            imgs_shape: The shape of the batch of images.
        """
        B, C, H, W = imgs_shape
        _, C_F, H_F, W_F = feature_maps.shape
        if torch.isnan(torch.concatenate(bboxes, dim=0)).any():
            raise ValueError("Bboxes contain NaNs; ROI Align will fail. This is a temporary failsafe.")
        H_ROI = bboxes[0][0,2] - bboxes[0][0,0] # just take 1st instance from 1st frame in batch
        W_ROI = bboxes[0][0,3] - bboxes[0][0,1]

        spatial_scale = (H_F/H + W_F/W)/2 # in case the scale isn't a round number

        # output_size = max(1, (H_ROI * spatial_scale).round().int()).item()
        # self.post_align_kernel_size = output_size
        return torchvision.ops.roi_align(feature_maps, bboxes, output_size=self.roi_align_output_size, spatial_scale=spatial_scale)
 
    def get_activation(self, name):
        """Get the activation of the layer."""
        def hook(model, input, output):
            self.layer_activation[name] = output.detach()
        return hook

    def forward(self, imgs: torch.Tensor, bboxes: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector.

        Args:
            imgs: Input image tensor of shape (B, C, H, W).
            bboxes: Input bounding box list of Tensor[num_instances, 4] for each frame.

        Returns:
            feats: Normalized output tensor of shape (B, d_model).
        """
        # If grayscale, tile the image to 3 channels.
        if imgs.shape[1] == 1:
            imgs = imgs.repeat([1, 3, 1, 1])  # (B, nc=3, H, W)

        if imgs.shape[1] != self.in_chans:
            raise ValueError(
                f"""Found {imgs.shape[1]} channels in image but model was configured for {self.in_chans} channels! \n
                    Hint: have you set the number of anchors in your dataset > 1? \n
                    If so, make sure to set `in_chans=3 * n_anchors`"""
            )
        # pass entire img through backbone to get the layer 4 feature map
        out_feat_vec = self.feature_extractor(
            imgs
        )  # (B, out_dim, 1, 1) if using resnet18 backbone.
        feature_maps = self.layer_activation['layer4'] # (B, 512, hf, wf) for resnet18; not necessarily square
        aligned_feature_maps = self.roi_align(feature_maps, bboxes, imgs.shape)

        aligned_feature_maps = self.post_align_conv1(aligned_feature_maps)
        aligned_feature_maps = self.bnorm1(aligned_feature_maps)
        aligned_feature_maps = F.relu(aligned_feature_maps)
        aligned_feature_maps = self.post_align_conv2(aligned_feature_maps)
        aligned_feature_maps = self.bnorm2(aligned_feature_maps)
        aligned_feature_maps = F.relu(aligned_feature_maps)
        aligned_feature_maps = aligned_feature_maps.squeeze()

        return aligned_feature_maps


class DescriptorVisualEncoder(torch.nn.Module):
    """Visual Encoder based on image descriptors."""

    def __init__(self, use_hu_moments: bool = False, **kwargs):
        """Initialize Descriptor Visual Encoder.

        Args:
            use_hu_moments: Whether to use Hu moments.
            **kwargs: Additional keyword arguments (unused but accepted for compatibility).
        """
        super().__init__()
        self.use_hu_moments = use_hu_moments

    def compute_hu_moments(self, img):
        """Compute Hu moments."""
        mu = measure.moments_central(img)
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)
        # log transform hu moments for scale differences; switched off; numerically unstable
        # hu_log = -np.sign(hu) * np.log(np.abs(hu))

        return hu

    def compute_inertia_tensor(self, img):
        """Compute inertia tensor."""
        return measure.inertia_tensor(img)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector."""
        descriptors = []

        for im in img:
            im = im[0].cpu().numpy()

            inertia_tensor = self.compute_inertia_tensor(im)
            mean_intensity = im.mean()
            if self.use_hu_moments:
                hu_moments = self.compute_hu_moments(im)

            # Flatten inertia tensor
            inertia_tensor_flat = inertia_tensor.flatten()

            # Combine all features into a single descriptor
            descriptor = np.concatenate(
                [
                    inertia_tensor_flat,
                    [mean_intensity],
                    hu_moments if self.use_hu_moments else [],
                ]
            )

            descriptors.append(torch.tensor(descriptor, dtype=torch.float32))

        return torch.stack(descriptors)


class DINOVisualEncoder(torch.nn.Module):
    """Visual Encoder based on DINO."""

    def __init__(self, d_model: int, use_pretrained: bool = True, **kwargs):
        """Initialize DINO Visual Encoder.

        Always uses pretrained models.

        Args:
            d_model: The dimension of the output feature vector.
            use_pretrained: Whether to use pretrained models (not supported yet).
            kwargs: Unused but accepted for compatibility
        """
        super().__init__()
        self.d_model = d_model
        # the pretrained models use patch_size=14
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        # else: # not currently supported due to dependency issues that it would cause
        #     self.model = dinov2.models.vision_transformer.vit_small(patch_size=14, num_register_tokens=4)
        self.mlp = torch.nn.Linear(self.model.num_features, d_model)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector."""
        if self.d_model != self.model.num_features:
            out = self.mlp(self.model(img))
        else:
            out = self.model(img)
        return out


def register_encoder(encoder_type: str, encoder_class: Type[torch.nn.Module]):
    """Register a new encoder type."""
    if not issubclass(encoder_class, torch.nn.Module):
        raise ValueError(f"{encoder_class} must be a subclass of torch.nn.Module")
    ENCODER_REGISTRY[encoder_type] = encoder_class


def create_visual_encoder(d_model: int, crop_size: int = None, **encoder_cfg) -> torch.nn.Module:
    """Create a visual encoder based on the specified type."""
    register_encoder("roi_align", VisualEncoderROIAlign)
    register_encoder("descriptor", DescriptorVisualEncoder)
    # register any custom encoders here
    register_encoder("dino", DINOVisualEncoder)

    # compatibility with configs that don't specify encoder_type; default to resnet
    if not encoder_cfg or "encoder_type" not in encoder_cfg:
        encoder_type = "roi_align"
        return ENCODER_REGISTRY[encoder_type](d_model=d_model, crop_size=crop_size, **encoder_cfg)
    else:
        encoder_type = encoder_cfg.pop("encoder_type")

    if encoder_type in ENCODER_REGISTRY:
        # choose the relevant encoder configs based on the encoder_type
        configs = encoder_cfg.pop("encoder_type_args", {})
        return ENCODER_REGISTRY[encoder_type](d_model=d_model, crop_size=crop_size, **configs)
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Please use one of {list(ENCODER_REGISTRY.keys())}"
        )
