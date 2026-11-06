"""Module for different visual feature extractors."""

import logging
from typing import Any, Dict, Type

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision

# import timm

# todo: do we want to make timm a dependency?
# todo: add named tensor support

logger = logging.getLogger("dreem.models")

ENCODER_REGISTRY: Dict[str, Type[torch.nn.Module]] = {}


class GeM(torch.nn.Module):
    """Generalized Mean (GeM) pooling over the spatial dimensions.

    Computes `(mean(x**p))**(1/p)` over `(H, W)`. `p = 1` recovers average
    pooling and `p -> inf` recovers max pooling; intermediate `p` (default 3)
    interpolates between them, emphasizing more salient activations
    (Radenovic et al., 2018, https://arxiv.org/abs/1711.02512).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        """Initialize GeM pooling.

        Args:
            p: Initial value of the pooling exponent.
            eps: Lower bound applied to activations before the power, for
                numerical stability.
            learnable: If True, `p` is a trainable parameter; otherwise it is a
                fixed buffer.
        """
        super().__init__()
        if learnable:
            self.p = torch.nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a feature map.

        Args:
            x: Feature map of shape `(B, C, H, W)`.

        Returns:
            Pooled tensor of shape `(B, C, 1, 1)`.
        """
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

    def extra_repr(self) -> str:
        """String representation of the pooling parameters."""
        p = self.p.data.item() if isinstance(self.p, torch.Tensor) else self.p
        return f"p={p:.4f}, eps={self.eps}"


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
        pooling: str = "gem",
        **kwargs: Any | None,
    ):
        """Initialize Visual Encoder.

        Args:
            model_name (str): Name of the CNN architecture to use (e.g. "resnet18", "resnet50").
            d_model (int): Output embedding dimension.
            in_chans: the number of input channels of the image.
            backend: Which model backend to use. One of {"timm", "torchvision"}
            pooling: Global pooling applied to the backbone feature map. One of
                {"gem", "avg", "max"}. Defaults to "gem" (learnable Generalized
                Mean pooling); "avg" reproduces the previous behavior.
            kwargs: see `timm.create_model` and `torchvision.models.resnetX` for kwargs.
        """
        super().__init__()

        self.model_name = model_name.lower()
        self.d_model = d_model
        self.backend = backend
        self.pooling = pooling.lower()
        if in_chans == 1:
            self.in_chans = 3
        else:
            self.in_chans = in_chans

        # Backbone emits an unpooled (B, C, H, W) feature map; pooling is applied
        # explicitly below so it is consistent across backends and configurable.
        self.feature_extractor = self.select_feature_extractor(
            model_name=self.model_name,
            in_chans=self.in_chans,
            backend=self.backend,
            **kwargs,
        )
        self.pool = self._build_pool(self.pooling)

        self.out_layer = torch.nn.Linear(
            self.encoder_dim(self.feature_extractor), self.d_model
        )

    @staticmethod
    def _build_pool(pooling: str) -> torch.nn.Module:
        """Build the global pooling module.

        Args:
            pooling: One of {"gem", "avg", "max"}.

        Returns:
            A module mapping `(B, C, H, W)` -> `(B, C, 1, 1)`.
        """
        if pooling == "gem":
            return GeM(p=3.0, learnable=True)
        if pooling == "avg":
            return torch.nn.AdaptiveAvgPool2d(1)
        if pooling == "max":
            return torch.nn.AdaptiveMaxPool2d(1)
        raise ValueError(
            f"`pooling` must be one of ['gem', 'avg', 'max'], found '{pooling}'"
        )

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
            # global_pool="" -> return the unpooled (B, C, H, W) feature map;
            # pooling is applied by self.pool in forward().
            kwargs.pop("global_pool", None)
            feature_extractor = timm.create_model(
                model_name=self.model_name,
                in_chans=self.in_chans,
                num_classes=0,
                global_pool="",
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
            # Drop both the classifier (fc) and the built-in avgpool so the
            # backbone returns the unpooled (B, C, H, W) feature map.
            feature_extractor = torch.nn.Sequential(
                *list(feature_extractor.children())[:-2]
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
        with torch.no_grad():
            dummy_output = self.pool(model(torch.randn(1, self.in_chans, 224, 224)))
        _ = model.train()  # to be safe
        return dummy_output.reshape(1, -1).shape[-1]

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
        feats = self.feature_extractor(img)  # (B, out_dim, H, W)
        feats = self.pool(feats)  # (B, out_dim, 1, 1)

        # Reshape feature vectors
        feats = feats.reshape([img.shape[0], -1])  # (B, out_dim)
        # Map feature vectors to output dimension using linear layer.
        feats = self.out_layer(feats)  # (B, d_model)
        # Normalize output feature vectors.
        feats = F.normalize(feats)  # (B, d_model)
        return feats


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
        import skimage.measure as measure

        mu = measure.moments_central(img)
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)
        # log transform hu moments for scale differences; switched off; numerically unstable
        # hu_log = -np.sign(hu) * np.log(np.abs(hu))

        return hu

    def compute_inertia_tensor(self, img):
        """Compute inertia tensor."""
        import skimage.measure as measure

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


class SleapNNVisualEncoder(torch.nn.Module):
    """A sleap-nn embedding model used as DREEM's visual encoder.

    Frozen by default; ``trainable=True`` fine-tunes it through DREEM's
    association loss instead (see below).

    DREEM's own encoder is trained end-to-end through the association loss, which
    never asks it to make an animal's crops similar across clips -- its positives
    span at most one chunk. A sleap-nn embedding model is trained with SupCon
    directly on identity, across sessions, so its features stay meaningful minutes
    or days later. This class swaps one for the other.

    Frozen means frozen
    -------------------
    Three separate things, all of which are required and none of which is implied
    by the others:

    1. ``requires_grad_(False)`` on every parameter, so no gradient is stored;
    2. the forward runs inside ``torch.no_grad()``, so no graph is built at all
       (this is what actually saves the memory -- (1) alone does not);
    3. ``train()`` is overridden to keep the submodule in eval mode. Lightning
       calls ``model.train()` at the start of every epoch, and without this the
       encoder's norm layers would drift even with `requires_grad=False`.

    The weights are still registered submodules, so they travel inside DREEM's
    checkpoint and inference needs no second file. That costs ~90 MB per
    checkpoint for a convnext_tiny backbone.

    Trainable
    ---------
    ``trainable=True`` undoes all three for the embedder (``encoder.model``):
    its parameters require grad, the forward builds a graph, and the encoder
    follows the parent's train/eval mode. sleap-nn's train-only SupCon
    ``projection`` head is not used by the forward and stays frozen. Its
    learning rate can be set apart from the rest of DREEM with
    ``optimizer.encoder_lr`` (see `GTRRunner.configure_optimizers`). A feature
    cache is refused in this mode (`dreem.training.train.run`): cached vectors
    bypass the encoder, so it would silently receive no gradient.

    Input conversion
    ----------------
    DREEM hands crops as ``(B, C, H, W)`` in ``[0, 1]`` (``SleapDataset`` divides
    by 255 when ``normalize_image``). sleap-nn's embedding model was trained on
    single-channel crops at ``[0, 255]`` and wants ``(B, 1, C, H, W)``. So:

    * RGB is collapsed with the ITU-R 601 luma weights -- the same weights
      ``make_sparse_crops.py`` used to build the crops the encoder trained on, so
      a colour DREEM crop lands where a grayscale training crop would have. An
      encoder trained on ONE channel instead (e.g. via sleap-io's
      ``grayscale=True`` video mode, which keeps channel 0, red) needs
      ``channel=0`` so DREEM's RGB crop is reduced the same way;
    * values are rescaled to ``[0, 255]``. ``EmbeddingLightningModule`` hardcodes
      ``self.standardize = True`` and standardizes per crop, so this is
      mathematically a no-op today -- it is done anyway so the input is right
      rather than accidentally right.

    ⚠ **Crop geometry is NOT checked here** and is the thing most likely to
    silently degrade this. sleap-nn's k-fold trained on crops cut with
    ``crop_centering: auto``; DREEM anchors on ``anchors``/``handle_missing``.
    Keep DREEM's ``crop_size`` equal to the encoder's and its anchor equivalent to
    the encoder's centering, or the frozen features are being asked about a
    different picture than they were trained on.

    Args:
        d_model: DREEM's model dimension. Must equal the encoder's embedding_dim
            unless ``project`` is set.
        checkpoint: path to the sleap-nn ``best.ckpt``.
        config: path to the run's ``training_config.yaml`` (the checkpoint does
            not carry the constructor arguments).
        project: add a trainable ``Linear(embed_dim, d_model)`` on top.
            Required when the dimensions differ.
        trainable: fine-tune the encoder instead of freezing it.
        channel: how a 3-channel crop is reduced to the single channel the
            encoder takes. ``None`` (default): ITU-R 601 luma. ``0``/``1``/``2``:
            keep only that channel -- match whatever produced the encoder's
            training crops.
    """

    def __init__(
        self,
        d_model: int = 512,
        checkpoint: str | None = None,
        config: str | None = None,
        project: bool = False,
        trainable: bool = False,
        channel: int | None = None,
        **kwargs,
    ):
        """Initialize the sleap-nn visual encoder."""
        super().__init__()
        if checkpoint is None or config is None:
            raise ValueError(
                "SleapNNVisualEncoder needs both `checkpoint` (best.ckpt) and "
                "`config` (training_config.yaml) in encoder_type_args."
            )
        # Imported lazily: sleap-nn is an optional extra (`pip install dreem[sleapnn]`)
        # and requires sleap-io >= 0.9.2, so it must not be imported by every run.
        try:
            from omegaconf import OmegaConf
            from sleap_nn.training.lightning_modules import EmbeddingLightningModule
        except ImportError as e:  # pragma: no cover - depends on optional extra
            # Report the original error: a sleap-nn that imports but lacks
            # EmbeddingLightningModule (any release <= 0.3.3) looks identical to
            # a missing install unless the cause is shown.
            raise ImportError(
                "encoder_type 'sleapnn' could not load sleap-nn's embedding "
                f"model ({e}). Install the extra with `uv sync --extra sleapnn`; "
                "note a released sleap-nn is NOT enough -- the embedding stack "
                "landed on main after 0.3.3."
            ) from e

        mc = OmegaConf.load(config).model_config
        backbone_type = [k for k, v in mc.backbone_config.items() if v is not None][0]
        self.encoder = EmbeddingLightningModule.load_from_checkpoint(
            checkpoint,
            map_location="cpu",
            model_type="embedding",
            backbone_type=backbone_type,
            backbone_config=mc.backbone_config,
            head_configs=mc.head_configs,
        )

        # head_configs.embedding.embedding.embedding_dim -- the head name is
        # nested inside its own group, matching sleap-nn's other heads.
        embed_dim = int(mc.head_configs.embedding.embedding.embedding_dim)
        self.d_model = d_model
        self.embed_dim = embed_dim
        if channel is not None and int(channel) not in (0, 1, 2):
            raise ValueError(f"channel must be None, 0, 1 or 2, got {channel!r}")
        self.channel = None if channel is None else int(channel)

        self.trainable = bool(trainable)
        if self.trainable:
            # Only `encoder.model` (backbone + EmbeddingHead) runs in forward.
            # Unfreeze it explicitly, overriding a sleap-nn `freeze_backbone`,
            # and freeze the train-only SupCon `projection` head, which never
            # receives a gradient here.
            self.encoder.requires_grad_(False)
            self.encoder.model.requires_grad_(True)
        else:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        if project:
            self.project = torch.nn.Linear(embed_dim, d_model)
        elif embed_dim != d_model:
            raise ValueError(
                f"sleap-nn embedding_dim ({embed_dim}) != d_model ({d_model}). "
                "Set encoder_type_args.project=true to learn a projection, or "
                "set model.d_model to match the encoder."
            )
        else:
            self.project = None

        state = "trainable" if self.trainable else "frozen"
        logger.info(
            f"SleapNNVisualEncoder: {state} {backbone_type} -> {embed_dim}d"
            + (f" -> Linear -> {d_model}d" if self.project is not None else "")
            + f"  ({sum(p.numel() for p in self.encoder.parameters()):,} {state} params)"
        )

    def _apply(self, fn, *args, **kwargs):
        """Keep the wrapped LightningModule's device bookkeeping in sync.

        `EmbeddingLightningModule.forward` does `img.to(self.device)`, and
        Lightning only updates `self.device` when `.to()` is called on that
        module itself. Moving a PARENT -- this wrapper, or Lightning moving the
        whole GTRRunner to the GPU -- goes through `nn.Module._apply` on the
        children, which relocates the weights but leaves `_device` reading cpu.
        The inner forward then moves the input back to CPU while its own weights
        are on CUDA, and the first conv raises a device mismatch.
        """
        out = super()._apply(fn, *args, **kwargs)
        param = next(self.encoder.parameters(), None)
        if param is not None:
            self.encoder._device = param.device
        return out

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval mode whatever the parent does.

        Lightning calls `model.train()` every epoch. Without this override the
        encoder's normalization layers would keep updating their running
        statistics, so a "frozen" encoder would quietly drift.
        """
        super().train(mode)
        if not self.trainable:
            self.encoder.eval()
        return self

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Crops -> normalized embeddings.

        Args:
            img: `(B, C, H, W)` crops in `[0, 1]`, C in {1, 3}.

        Returns:
            `(B, d_model)` embeddings, L2-normalized (matching `VisualEncoder`).
        """
        if img.ndim != 4:
            raise ValueError(f"expected (B, C, H, W) crops, got {tuple(img.shape)}")

        if img.shape[1] == 3 and self.channel is not None:
            gray = img[:, self.channel : self.channel + 1]
        elif img.shape[1] == 3:
            # ITU-R 601 luma, matching make_sparse_crops.py's PIL "L" conversion.
            w = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype)
            gray = (img * w[None, :, None, None]).sum(dim=1, keepdim=True)
        elif img.shape[1] == 1:
            gray = img
        else:
            raise ValueError(f"expected 1 or 3 channels, got {img.shape[1]}")

        gray = gray * 255.0                       # back to the training scale
        x = gray.unsqueeze(1)                     # (B, 1, C, H, W) as forward() wants

        param = next(self.encoder.parameters(), None)
        if param is not None and x.device != param.device:
            x = x.to(param.device)

        if self.trainable:
            feats = self.encoder(x)
        else:
            with torch.no_grad():
                feats = self.encoder(x)

        if self.project is not None:
            feats = self.project(feats)
            feats = F.normalize(feats)
        return feats


class NullVisualEncoder(torch.nn.Module):
    """Visual encoder that returns zeros, so association sees only position and time.

    Asks how much DREEM can do from its positional and temporal embeddings alone.
    It has no parameters and never reads the crop, so a dataset can hand it a 1x1
    placeholder instead of decoding frames (`SleapDataset(skip_image=True)`).
    Under concatenate fusion the visual slice becomes `visual_proj.bias`, the same
    constant for every instance; under average fusion the queries are the
    embeddings alone.
    """

    def __init__(self, d_model: int, **kwargs):
        """Initialize the null encoder.

        Args:
            d_model: Output feature dimension, matching the transformer.
            **kwargs: Accepted and ignored, so any encoder config can be swapped in.
        """
        super().__init__()
        self.d_model = d_model
        # Tracks the module's device, so zeros land beside the transformer
        # even if the placeholder crops were left on the CPU.
        self.register_buffer("_anchor", torch.zeros(0), persistent=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Return a (B, d_model) block of zeros, one row per crop."""
        return torch.zeros(img.shape[0], self.d_model, device=self._anchor.device)


def register_encoder(encoder_type: str, encoder_class: Type[torch.nn.Module]):
    """Register a new encoder type."""
    if not issubclass(encoder_class, torch.nn.Module):
        raise ValueError(f"{encoder_class} must be a subclass of torch.nn.Module")
    ENCODER_REGISTRY[encoder_type] = encoder_class


def create_visual_encoder(d_model: int, **encoder_cfg) -> torch.nn.Module:
    """Create a visual encoder based on the specified type."""
    register_encoder("resnet", VisualEncoder)
    register_encoder("descriptor", DescriptorVisualEncoder)
    # register any custom encoders here
    register_encoder("dino", DINOVisualEncoder)
    register_encoder("sleapnn", SleapNNVisualEncoder)
    register_encoder("none", NullVisualEncoder)

    # compatibility with configs that don't specify encoder_type; default to resnet
    if not encoder_cfg or "encoder_type" not in encoder_cfg:
        encoder_type = "resnet"
        return ENCODER_REGISTRY[encoder_type](d_model=d_model, **encoder_cfg)
    else:
        encoder_type = encoder_cfg.pop("encoder_type")

    if encoder_type in ENCODER_REGISTRY:
        # choose the relevant encoder configs based on the encoder_type
        configs = encoder_cfg.pop("encoder_type_args", {})
        return ENCODER_REGISTRY[encoder_type](d_model=d_model, **configs)
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Please use one of {list(ENCODER_REGISTRY.keys())}"
        )
