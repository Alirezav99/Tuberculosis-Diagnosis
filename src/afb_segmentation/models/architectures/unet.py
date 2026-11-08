"""U-Net layers and mask-prediction architectures.

The module defines standard and attention-gated encoder-decoder networks. Both
model heads return sigmoid probabilities, and convolution blocks use batch
normalization with ReLU activation. Optional dropout is applied in the
convolution blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net, with optional dropout."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        """Initialize two convolution-normalization-activation layers.

        Parameters
        ----------
        in_channels
            Number of input feature channels.
        out_channels
            Number of output feature channels.
        mid_channels
            Intermediate channel count; defaults to
            ``out_channels`` when falsey.
        dropout
            Dropout2d probability inserted after each activation when
            greater than zero.
        """
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        """Apply the convolution block to a ``(B, C, H, W)`` feature tensor."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv (with dropout)"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        """Initialize max pooling followed by a ``DoubleConv`` block."""
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        """Downsample and transform the input feature map."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv (with dropout)"""

    def __init__(self, in_channels, out_channels, bilinear=False, dropout=0.0):
        """Initialize the upsampling and convolution stages.

        Parameters
        ----------
        in_channels
            Decoder input channel count.
        out_channels
            Output channel count after the convolution block.
        bilinear
            Use bilinear interpolation when true; otherwise use a
            transposed convolution.
        dropout
            Dropout2d probability for the convolution block.
        """
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, dropout=dropout
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        """Upsample decoder features, align them, concatenate the skip, and convolve.

        Parameters
        ----------
        x1
            Decoder feature map to upsample.
        x2
            Encoder feature map used as the skip connection.

        Returns
        -------
        result : torch.Tensor
            The fused feature map.
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""

    def __init__(self, in_channels, out_channels):
        """Initialize a 1-by-1 convolution from input to output channels."""
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Project the feature map to the requested output channels."""
        return self.conv(x)


class UNetMaskPredictor(nn.Module):
    """U-Net model for mask prediction

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images)
    out_channels : int
        Number of output channels (e.g., 1 for binary mask)
    bilinear : bool
        If True, use bilinear upsampling instead of transposed convolutions
    dropout : float
        Dropout rate applied in convolution blocks
    base_channels : int
        Base number of channels in the first layer (default: 64)
    depth : int
        Number of downsampling/upsampling stages (default: 4)
    channel_factor : int
        Multiplier for channels at each depth level (default: 2)
    scale_factor : int
        Divides all channel counts for a lighter model (default: 1)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        bilinear: bool = False,
        dropout: float = 0.0,
        base_channels: int = 64,
        depth: int = 4,
        channel_factor: int = 2,
        scale_factor: int = 1,
    ):
        """Build the encoder, decoder, skip connections, and sigmoid head.

        Parameters
        ----------
        in_channels : int
            Number of input image channels.
        out_channels : int
            Number of output mask channels.
        bilinear : bool
            Select bilinear upsampling instead of transposed
            convolutions in decoder blocks.
        dropout : float
            Dropout2d probability in convolution blocks.
        base_channels : int
            Channel count at the first encoder level.
        depth : int
            Number of downsampling and upsampling stages.
        channel_factor : int
            Per-level channel multiplier.
        scale_factor : int
            Divisor applied to the computed channel counts.
        """
        super(UNetMaskPredictor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.dropout = dropout
        self.base_channels = base_channels
        self.depth = depth
        self.channel_factor = channel_factor
        self.scale_factor = scale_factor

        self.channels = [
            base_channels * (channel_factor**i) // scale_factor
            for i in range(depth + 1)
        ]

        self.inc = DoubleConv(in_channels, self.channels[0], dropout=dropout)

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(
                Down(self.channels[i], self.channels[i + 1], dropout=dropout)
            )

        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(
                Up(self.channels[i], self.channels[i - 1], bilinear, dropout=dropout)
            )

        self.outc = OutConv(self.channels[0], out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return sigmoid mask probabilities for an image batch."""

        encoder_features = [self.inc(x)]
        for down in self.downs:
            encoder_features.append(down(encoder_features[-1]))

        x = encoder_features[-1]
        for i, up in enumerate(self.ups):
            skip = encoder_features[-(i + 2)]
            x = up(x, skip)

        logits = self.outc(x)
        return self.sigmoid(logits)


class AttentionGate(nn.Module):
    """Attention Gate for U-Net skip connections.

    Learns to focus on relevant features (bacilli) while suppressing irrelevant
    regions (staining artifacts, background). The gating signal from the decoder
    provides contextual information to selectively highlight encoder features.

    Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
    arXiv:1804.03999

    Parameters
    ----------
    F_g : int
        Number of channels in gating signal (from decoder)
    F_l : int
        Number of channels in encoder feature map (skip connection)
    F_int : int
        Number of intermediate channels for attention computation
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """Initialize projections for decoder gating and encoder skip features.

        Parameters
        ----------
        F_g : int
            Number of channels in the decoder gating signal.
        F_l : int
            Number of channels in the encoder skip feature.
        F_int : int
            Intermediate attention channel count.
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        g : torch.Tensor
            Gating signal from decoder path (lower resolution, more semantic)
        x : torch.Tensor
            Encoder feature map from skip connection (higher resolution)

        Returns
        -------
        result : torch.Tensor
            Attention-weighted encoder features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=True
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUp(nn.Module):
    """Upsample decoder features and gate encoder features before fusion."""

    def __init__(self, in_channels, out_channels, bilinear=False, dropout=0.0):
        """Initialize upsampling, attention gating, and feature fusion.

        Parameters
        ----------
        in_channels
            Decoder input channel count.
        out_channels
            Output channel count after feature fusion.
        bilinear
            Select bilinear interpolation instead of transposed
            convolution.
        dropout
            Dropout2d probability in the convolution block.
        """
        super(AttentionUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, dropout=dropout
            )

            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

            up_channels = in_channels // 2

        skip_channels = in_channels // 2
        self.attention = AttentionGate(
            F_g=up_channels, F_l=skip_channels, F_int=skip_channels // 2
        )

    def forward(self, x1, x2):
        """Upsample decoder input and fuse it with an attention-weighted skip.

        Parameters
        ----------
        x1
            Decoder feature map (to be upsampled)
        x2
            Encoder feature map (skip connection)

        Returns
        -------
        result : torch.Tensor
            The fused feature map after convolution.
        """
        x1 = self.up(x1)

        x2 = self.attention(g=x1, x=x2)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNetMaskPredictor(nn.Module):
    """Attention U-Net for TB bacilli mask prediction.

    Enhanced U-Net with attention gates at each skip connection to focus on
    relevant bacilli features while suppressing staining artifacts and background.

    Key improvements over standard U-Net:
    - Attention gates learn to highlight rod-shaped bacilli structures
    - Suppresses false positives from fuchsia staining debris
    - Better boundary delineation for accurate morphology

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images)
    out_channels : int
        Number of output channels (e.g., 1 for binary mask)
    bilinear : bool
        If True, use bilinear upsampling instead of transposed convolutions
    dropout : float
        Dropout rate applied in convolution blocks
    base_channels : int
        Base number of channels in the first layer (default: 64)
    depth : int
        Number of downsampling/upsampling stages (default: 4)
    channel_factor : int
        Multiplier for channels at each depth level (default: 2)
    scale_factor : int
        Divides all channel counts for a lighter model (default: 1)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        bilinear: bool = False,
        dropout: float = 0.0,
        base_channels: int = 64,
        depth: int = 4,
        channel_factor: int = 2,
        scale_factor: int = 1,
    ):
        """Build an attention-gated U-Net for mask probability prediction.

        Parameters
        ----------
        in_channels : int
            Number of input image channels.
        out_channels : int
            Number of output mask channels.
        bilinear : bool
            Select bilinear decoder upsampling when true.
        dropout : float
            Dropout2d probability in convolution blocks.
        base_channels : int
            Channel count at the first encoder level.
        depth : int
            Number of downsampling and upsampling stages.
        channel_factor : int
            Per-level channel multiplier.
        scale_factor : int
            Divisor applied to the computed channel counts.
        """
        super(AttentionUNetMaskPredictor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.dropout = dropout
        self.base_channels = base_channels
        self.depth = depth
        self.channel_factor = channel_factor
        self.scale_factor = scale_factor

        self.channels = [
            base_channels * (channel_factor**i) // scale_factor
            for i in range(depth + 1)
        ]

        self.inc = DoubleConv(in_channels, self.channels[0], dropout=dropout)

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(
                Down(self.channels[i], self.channels[i + 1], dropout=dropout)
            )

        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(
                AttentionUp(
                    self.channels[i], self.channels[i - 1], bilinear, dropout=dropout
                )
            )

        self.outc = OutConv(self.channels[0], out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return sigmoid mask probabilities using attention-gated skip features."""

        encoder_features = [self.inc(x)]
        for down in self.downs:
            encoder_features.append(down(encoder_features[-1]))

        x = encoder_features[-1]
        for i, up in enumerate(self.ups):
            skip = encoder_features[-(i + 2)]
            x = up(x, skip)

        logits = self.outc(x)
        return self.sigmoid(logits)
