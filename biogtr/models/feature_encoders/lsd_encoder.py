"""Module containing parts necessary for generating LSD embeddings as features and for auxiliary learning task."""
import torch
import math
from torch import nn
from biogtr.models.feature_encoders.base_feature_extractor import BaseFeatureExtractor


class LSDEncoder(BaseFeatureExtractor):
    """Class containing encoder for LSDs.

    In the auxiliary learning setup, encoder predicts LSDs and thus is a UNet with raw image input.
    Otherwise, encoder simply takes in computed LSD as input, flattens it and projects it to correct dimensionality.
    """

    def __init__(self, unet_cfg: dict = {}, d_model: int = 512, normalize: bool = True):
        """Initialize LSD Encoder.

        Args:
            unet_cfg: dictionary containing hyperparameters for UNNet.
            If dictionary is None, then UNet is simply the Identity function. Otherwise we assume the auxiliary learning setting.
            d_model: dimension of subspace to project features into.
            normalize: Whether to normalize after projection or not.
        """
        super().__init__(d_model, normalize)

        if unet_cfg is not None:
            self.unet = UNet(**unet_cfg)
        else:
            self.unet = torch.nn.Identity()

    def _forward_features(self, x: torch.TensorType) -> torch.TensorType:
        """Extract LSD Embedding.

        In the auxiliary learning setup, LSDs are predicted by a unnet then projected into an embedding.
        Otherwise, the LSDs are fed directly into the projector to get out an embedding.

        Args:
            x: A tensor containing either the raw image (shape is [B, C, H, W]) or a tensor containing the precomputed LSDs (shape is [B, 6, H, W])

        Returns:
            an LSD embedding of shape (B, 6HW)
        """
        if isinstance(self.unet, UNet) and x.shape[1] >= 6:
            raise ValueError(
                "It appears you passed the lsds in instead of the raw image! \
                             Encoder is currently configured for auxiliary learning task."
            )
        if isinstance(self.unet, torch.nn.Identity) and x.shape[1] <= 6:
            raise ValueError(
                "It appears you passed the raw image in instead of lsds! \
                             Encoder is currently not configured for auxiliary learning task."
            )
        x = self.unet(x)
        print(x.shape)
        return x.view(x.size(0), -1)


class UNet(torch.nn.Module):
    """Implementation of a U-Net architecture for predicting LSDs.

    Attributes:
            num_levels (int): Number of U-Net levels.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size_down (list of lists): List specifying kernel sizes for downsample convolutional passes.
            kernel_size_up (list of lists): List specifying kernel sizes for upsample convolutional passes.
            downsample_factors (list of tuples): List specifying downsampling factors at each level.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=6,
        num_fmaps=32,
        fmap_inc_factors=3,
        downsample_factors=[(2, 2), (2, 2)],
        kernel_size_down=None,
        kernel_size_up=None,
        activation="ReLU",
        num_fmaps_out=None,
    ):
        """Initialize U-Net.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            out_channels (int, optional): Number of output channels. Defaults to 6.
            num_fmaps (int, optional): Number of feature maps. Defaults to 32.
            fmap_inc_factors (int, optional): Factor by which the number of feature maps
                                            increases with each level. Defaults to 3.
            downsample_factors (list of tuples, optional): List of tuples specifying the downsampling
                                                        factors at each level. Defaults to [(2, 2), (2, 2)].
            kernel_size_down (list of lists, optional): List of lists specifying the kernel sizes for
                                                        downsample convolutional passes at each level.
                                                        Defaults to [[(3, 3), (3, 3)]] * num_levels.
            kernel_size_up (list of lists, optional): List of lists specifying the kernel sizes for
                                                    upsample convolutional passes at each level (excluding
                                                    the last level). Defaults to [[(3, 3), (3, 3)]] * (num_levels - 1).
            activation (str, optional): Activation function to use. Defaults to "ReLU".
            num_fmaps_out (int, optional): Number of output feature maps. If not provided, uses num_fmaps.
                                        Defaults to None.
        """
        super(UNet, self).__init__()

        self.num_levels = len(downsample_factors) + 1
        self.in_channels = in_channels
        self.out_channels = num_fmaps_out if num_fmaps_out else num_fmaps

        if kernel_size_down is None:
            kernel_size_down = [[(3, 3), (3, 3)]] * self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3, 3), (3, 3)]] * (self.num_levels - 1)

        self.kernel_size_down = kernel_size_down
        self.kernel_size_up = kernel_size_up
        self.downsample_factors = downsample_factors

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(f * ff for f, ff in zip(factor, factor_product))
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList(
            [
                ConvPass(
                    in_channels
                    if level == 0
                    else num_fmaps * fmap_inc_factors ** (level - 1),
                    num_fmaps * fmap_inc_factors**level,
                    kernel_size_down[level],
                    activation=activation,
                )
                for level in range(self.num_levels)
            ]
        )
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList(
            [
                Downsample(downsample_factors[level])
                for level in range(self.num_levels - 1)
            ]
        )

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList(
            [
                Upsample(
                    downsample_factors[level],
                    in_channels=num_fmaps * fmap_inc_factors ** (level + 1),
                    out_channels=num_fmaps * fmap_inc_factors ** (level + 1),
                    crop_factor=crop_factors[level],
                    next_conv_kernel_sizes=kernel_size_up[level],
                )
                for level in range(self.num_levels - 1)
            ]
        )

        # right convolutional passes
        self.r_conv = nn.ModuleList(
            [
                ConvPass(
                    num_fmaps * fmap_inc_factors**level
                    + num_fmaps * fmap_inc_factors ** (level + 1),
                    num_fmaps * fmap_inc_factors**level
                    if num_fmaps_out is None or level != 0
                    else num_fmaps_out,
                    kernel_size_up[level],
                    activation=activation,
                )
                for level in range(self.num_levels - 1)
            ]
        )

        self.final_conv = ConvPass(
            in_channels=num_fmaps,
            out_channels=out_channels,
            kernel_sizes=[(1, 1), (1, 1)],
            activation="Sigmoid",
        )

    def rec_forward(self, level: int, f_in: torch.Tensor) -> torch.Tensor:
        """Recursively perform forward pass through U-Net architecture at a specified level.

        This method recursively performs a forward pass through the U-Net architecture at the specified level.
        The input tensor `f_in` is convolved at the current level, and if the recursion has not reached the base
        level (level 0), it continues by downsampling, performing nested levels, and then upsampling, concatenating,
        and cropping the results. The final tensor is obtained by convolving the concatenated and cropped tensor.
        The output tensor represents the result of the forward pass at the specified level.

        Args:
            level (int): Current U-Net level during recursion.
            f_in (torch.Tensor): Input tensor at the current level.

        Returns:
            torch.Tensor: Output tensor after recursive forward pass.
        """
        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:
            fs_out = f_left
        else:
            # down
            g_in = self.l_down[i](f_left)
            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)
            # up, concat, and crop
            fs_right = self.r_up[i](f_left, gs_out)

            # convolve
            fs_out = self.r_conv[i](fs_right)

        return fs_out

    def forward(self, x):
        """Forward pass through the U-Net architecture.

        This method performs a forward pass through the U-Net architecture by calling the recursive forward
        pass (`rec_forward`) starting from the top level. The output of the recursive pass is then processed
        through the final convolutional layer. The resulting tensor represents the output of the U-Net for
        the given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        y = self.rec_forward(self.num_levels - 1, x)

        out = self.final_conv(y)

        return out


class ConvPass(torch.nn.Module):
    """Convolutional pass module.

    Attributes:
        dims (int): Number of dimensions (e.g., 2 for Conv2d).
        conv_pass (torch.nn.Sequential): Sequential container for convolutional layers and activation functions.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, activation):
        """Initialize Convolutional pass module with optional activation function.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list of tuples): List of kernel sizes for convolutional layers.
            activation (str): Activation function to apply after each convolution. If None, no activation is applied.
        """
        super(ConvPass, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)

        layers = []

        for kernel_size in kernel_sizes:
            self.dims = len(kernel_size)

            pad = tuple(torch.tensor(kernel_size) // 2)

            layers.append(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
            )
            in_channels = out_channels

            if activation is not None:
                layers.append(activation())
                # layers.append(nn.BatchNorm2d(out_channels))

        self.conv_pass = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the conv layer.

        Args:
            x: input tensor.

        Returns:
            output tensor.
        """
        return self.conv_pass(x)


class Downsample(torch.nn.Module):
    """Downsample module using max pooling.

    Attributes:
        dims (int): Number of dimensions (e.g., 2 for MaxPool2d).
        downsample_factor (tuple): Tuple specifying the downsampling factors for each dimension.
        down (torch.nn.MaxPool2d): Max pooling layer for downsampling.
    """

    def __init__(self, downsample_factor: tuple):
        """Initialize Downsample module.

        Args:
            downsample_factor (tuple): Tuple specifying the downsampling factors for each dimension.
        """
        super(Downsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor

        self.down = torch.nn.MaxPool2d(downsample_factor, stride=downsample_factor)

    def forward(self, x):
        """Forward pass through the Downsample module using max pooling.

        This method performs a forward pass through the Downsample module by applying max pooling to
        downsample the input tensor 'x'. It ensures that the spatial dimensions of 'x' are divisible
        by the specified downsampling factors. If not, a RuntimeError is raised.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after max pooling downsampling.
        """
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (x.size(), self.downsample_factor, self.dims - d)
                )

        return self.down(x)


class Upsample(torch.nn.Module):
    """Upsample module using nearest-neighbor interpolation.

    Attributes:
        dims (int): Number of dimensions (e.g., 2 for Upsample).
        crop_factor (tuple): Tuple specifying the crop factor for each dimension.
        next_conv_kernel_sizes (list of tuples): List of kernel sizes for the next convolutional layers.
        up (torch.nn.Upsample): Upsample layer using nearest-neighbor interpolation.
    """

    def __init__(
        self,
        scale_factor,
        in_channels=None,
        out_channels=None,
        crop_factor=None,
        next_conv_kernel_sizes=None,
    ):
        """Initialize Upsample module with nearest-neighbor interpolation.

        Args:
            scale_factor (tuple): Tuple specifying the scaling factors for each dimension.
            in_channels (int, optional): Number of input channels. Defaults to None.
            out_channels (int, optional): Number of output channels. Defaults to None.
            crop_factor (tuple, optional): Tuple specifying the crop factor for each dimension.
            next_conv_kernel_sizes (list of tuples, optional): List of kernel sizes for the next convolutional layers.
        """
        super(Upsample, self).__init__()

        assert (crop_factor is None) == (
            next_conv_kernel_sizes is None
        ), "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes

        self.dims = len(scale_factor)

        self.up = torch.nn.Upsample(scale_factor=tuple(scale_factor), mode="nearest")

    def crop_to_factor(
        self, x: torch.Tensor, factor: tuple, kernel_sizes: list[tuple]
    ) -> torch.Tensor:
        """Crop the input tensor to ensure translation equivariance with specified factor and convolutions.

        This method calculates the target spatial shape based on the input tensor, convolutional kernel sizes,
        and the specified factor. It then crops the input tensor to ensure translation equivariance with the factor
        and subsequent convolutions. If the target spatial shape cannot be achieved, a RuntimeError is raised.

        Args:
            x (torch.Tensor): Input tensor.
            factor (tuple): Tuple specifying the factor for each dimension.
            kernel_sizes (list of tuples): List of kernel sizes for the following convolutions.

        Returns:
            torch.Tensor: Cropped output tensor.
        """
        shape = x.size()
        spatial_shape = shape[-self.dims :]

        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes) for d in range(self.dims)
        )

        ns = (
            int(math.floor(float(s - c) / f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n * f + c for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:
            assert all(
                ((t > c) for t, c in zip(target_spatial_shape, convolution_crop))
            ), (
                "Feature map with shape %s is too small to ensure "
                "translation equivariance with factor %s and following "
                "convolutions %s" % (shape, factor, kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """Crop the input tensor to the specified shape.

        This method crops the input tensor to the specified target shape. The cropping is performed by calculating
        the offset and creating slices for each dimension. The resulting tensor represents the cropped output.

        Args:
            x (torch.Tensor): Input tensor.
            shape (tuple): Target shape for cropping.

        Returns:
            torch.Tensor: Cropped output tensor.
        """
        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        sliced = x[slices]

        return sliced

    def forward(self, f_left, g_out):
        """Forward pass through the Upsample module.

        This method performs a forward pass through the Upsample module by upsampling the tensor from
        the lower branch ('g_out') using nearest-neighbor interpolation. The upsampled tensor is then cropped
        to match the spatial dimensions of the input tensor from the left branch ('f_left'). Finally, the
        cropped tensors are concatenated along the channel dimension and returned as the output.

        Args:
            f_left (torch.Tensor): Input tensor from the left branch.
            g_out (torch.Tensor): Output tensor from the lower branch.

        Returns:
            torch.Tensor: Concatenated and cropped output tensor.
        """
        g_up = self.up(g_out)

        g_cropped = g_up

        f_cropped = self.crop(f_left, g_cropped.size()[-self.dims :])

        return torch.cat([f_cropped, g_cropped], dim=1)
