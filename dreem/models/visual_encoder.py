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

def select_feature_extractor(
        model_name: str, in_chans: int, backend: str, **kwargs: Any
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
                model_name=model_name,
                in_chans=in_chans,
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

def encoder_dim(in_chans: int, model: torch.nn.Module) -> int:
        """Compute dummy forward pass of encoder model and get embedding dimension.

        Args:
            model: a vision encoder model.

        Returns:
            The embedding dimension size.
        """
        _ = model.eval()
        dummy_output = model(torch.randn(1, in_chans, 224, 224)).squeeze()
        _ = model.train()  # to be safe
        return dummy_output.shape[-1]


class VisualEncoder(torch.nn.Module):
    """Class wrapping around a visual feature extractor backbone.

    Currently CNN only.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        d_model: int = 512,
        in_chans: int = 3,
        backend: int = "timm",
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
        if in_chans == 1:
            self.in_chans = 3
        else:
            self.in_chans = in_chans

        self.feature_extractor = select_feature_extractor(
            model_name=self.model_name,
            in_chans=self.in_chans,
            backend=self.backend,
            **kwargs,
        )

        self.out_layer = torch.nn.Linear(
            encoder_dim(self.in_chans, self.feature_extractor), self.d_model
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            feats: Normalized output tensor of shape (B, d_model).
        """
        # If grayscale, tile the image to 3 channels.
        if img.shape[1] == 1:
            img = img.repeat([1, 3, 1, 1])  # (B, nc=3, H, W)

        b, c, h, w = img.shape

        if c != self.in_chans:
            raise ValueError(
                f"""Found {c} channels in image but model was configured for {self.in_chans} channels! \n
                    Hint: have you set the number of anchors in your dataset > 1? \n
                    If so, make sure to set `in_chans=3 * n_anchors`"""
            )
        feats = self.feature_extractor(
            img
        )  # (B, out_dim, 1, 1) if using resnet18 backbone.

        # Reshape feature vectors
        feats = feats.reshape([img.shape[0], -1])  # (B, out_dim)
        # Map feature vectors to output dimension using linear layer.
        feats = self.out_layer(feats)  # (B, d_model)
        # Normalize output feature vectors.
        feats = F.normalize(feats)  # (B, d_model)
        return feats



def roi_align(feature_maps: list[torch.Tensor], bboxes: list[torch.Tensor], imgs_shape: torch.Tensor, roi_align_output_size: int) -> torch.Tensor:
    """Roi align the feature map.

    Args:
        feature_maps: List of feature maps to align of shape (B, C, H_F, W_F).
        bboxes: Input bounding box list of Tensor[num_instances, 4] for each frame.
        imgs_shape: The shape of the batch of images.
        roi_align_output_size: The output size of the ROI Align.
    """
    B, C, H, W = imgs_shape
    if torch.isnan(torch.concatenate(bboxes, dim=0)).any():
        raise ValueError("Bboxes contain NaNs; ROI Align will fail. This is a temporary failsafe.")
    # choose ROI level
    img_area = H*W
    h_roi = bboxes[0][0][3] - bboxes[0][0][1] # bboxes is a list of (num_instances, 4)
    w_roi = bboxes[0][0][2] - bboxes[0][0][0]
    roi_level = torch.log(torch.sqrt(h_roi*w_roi)/torch.sqrt(torch.tensor(img_area)))
    roi_level = torch.round(roi_level + 4)
    roi_level[roi_level < 3] = 3
    roi_level[roi_level > 5] = 5
    roi_level = roi_level.squeeze().int().item()
    feature_map = feature_maps[roi_level - len(feature_maps)]
    spatial_scale = feature_map.shape[2] / H # how much to scale the bbox coords to match feature map size

    return torchvision.ops.roi_align(feature_map, bboxes, output_size=roi_align_output_size, spatial_scale=spatial_scale)

def get_activation(name, dict_activation):
        """Get the activation of the layer."""
        def hook(model, input, output):
            dict_activation[name] = output.detach()
        return hook

class ROIAlignVisualEncoder(torch.nn.Module):
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

        self.feature_extractor = select_feature_extractor(
            model_name=self.model_name,
            in_chans=self.in_chans,
            backend=self.backend,
            **kwargs,
        )
        self.layer_activation = {}
        self.feature_extractor.layer4.register_forward_hook(get_activation('layer4', self.layer_activation))
        self.feature_extractor.layer3.register_forward_hook(get_activation('layer3', self.layer_activation))
        self.feature_extractor.layer2.register_forward_hook(get_activation('layer2', self.layer_activation))
        # layer 2 is downsampled 8x so a crop of size 32 would become 4x4

        self.latlayer1 = torch.nn.Conv2d(self.feature_extractor.layer4[-1].conv2.out_channels, 256, (1, 1), stride=1, padding=0) # 512
        self.latlayer2 = torch.nn.Conv2d(self.feature_extractor.layer3[-1].conv2.out_channels, 256, (1, 1), stride=1, padding=0) # 256
        self.latlayer3 = torch.nn.Conv2d(self.feature_extractor.layer2[-1].conv2.out_channels, 256, (1, 1), stride=1, padding=0) # 128
        
        # TODO: fix hardcoded value. 4x4 is the smallest a crop would map to on the feature map assuming min crop size of 32
        self.roi_align_output_size = 4
        print("ROI Align output map size: ", self.roi_align_output_size)
        self.post_align_conv1 = torch.nn.Conv2d(256, d_model, (self.roi_align_output_size, self.roi_align_output_size), stride=1, padding=0)
        self.bnorm1 = torch.nn.BatchNorm2d(d_model)
        self.post_align_conv2 = torch.nn.Conv2d(d_model, d_model, (1, 1), stride=1, padding=0)
        self.bnorm2 = torch.nn.BatchNorm2d(d_model)

    def forward(self, imgs: torch.Tensor, bboxes: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass of feature extractor to get feature vector.

        Args:
            imgs: Input image tensor of shape (B, C, H, W).
            bboxes: Input bounding box list of Tensor[N, 4] for each frame.

        Returns:
            feats: Normalized output tensor of shape (N, d_model), where N is the number of instances in the batch
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
        out = self.feature_extractor(
            imgs
        )  # TODO: since not using output, remove the out layers to avoid unnecessary compute
        c3 = self.layer_activation['layer2'] # (B, 128, hf, wf) for resnet18; not necessarily square
        c4 = self.layer_activation['layer3'] # (B, 256, hf, wf) for resnet18; not necessarily square
        c5 = self.layer_activation['layer4'] # (B, 512, hf, wf) for resnet18; not necessarily square
        p5 = self.latlayer1(c5) # latlayer1 is for layer 4 output
        p4 = upsample_add(p5, self.latlayer2(c4))
        p3 = upsample_add(p4, self.latlayer3(c3))
        feature_maps = [p3, p4, p5]

        aligned_feature_maps = roi_align(feature_maps, bboxes, imgs.shape, self.roi_align_output_size) # (N, 256, output, output)
        self.layer_activation.clear()
        del c3, c4, c5, p3, p4, p5, feature_maps, out
        torch.cuda.empty_cache()

        aligned_feature_maps = self.post_align_conv1(aligned_feature_maps)
        aligned_feature_maps = self.bnorm1(aligned_feature_maps)
        aligned_feature_maps = F.relu(aligned_feature_maps)
        aligned_feature_maps = self.post_align_conv2(aligned_feature_maps)
        aligned_feature_maps = self.bnorm2(aligned_feature_maps)
        aligned_feature_maps = F.relu(aligned_feature_maps)
        aligned_feature_maps = aligned_feature_maps.squeeze() # (N, d_model)

        return aligned_feature_maps


def upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample and add two feature maps.

        Args:
          x: (Tensor) top feature map to be upsampled.
          y: (Tensor) lateral feature map.

        Returns:
          (Tensor) added feature map.
        """
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y


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


def register_encoder(encoder_type: str, encoder_class: Type[torch.nn.Module]):
    """Register a new encoder type."""
    if not issubclass(encoder_class, torch.nn.Module):
        raise ValueError(f"{encoder_class} must be a subclass of torch.nn.Module")
    ENCODER_REGISTRY[encoder_type] = encoder_class


def create_visual_encoder(d_model: int, crop_size: int | None = None, **encoder_cfg) -> torch.nn.Module:
    """Create a visual encoder based on the specified type."""
    register_encoder("resnet", VisualEncoder)
    register_encoder("descriptor", DescriptorVisualEncoder)
    # register any custom encoders here
    register_encoder("roi_align", ROIAlignVisualEncoder)

    # compatibility with configs that don't specify encoder_type; default to resnet
    if not encoder_cfg or "encoder_type" not in encoder_cfg:
        encoder_type = "resnet"
        return ENCODER_REGISTRY[encoder_type](d_model=d_model, **encoder_cfg)
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
