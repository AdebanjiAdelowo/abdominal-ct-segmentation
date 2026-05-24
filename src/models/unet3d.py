"""
3D U-Net for volumetric medical image segmentation.

Architecture based on Çiçek et al. (2016) with the following modifications:
  - Instance normalisation instead of batch normalisation (stable for small
    batch sizes common in 3-D training).
  - LeakyReLU activations (slope 0.01) to prevent dead neurons.
  - Optional residual shortcut within each ConvBlock.
  - Configurable depth: feature widths double at each encoder level.
  - Kaiming weight initialisation tuned for LeakyReLU.

Reference:
    Ö. Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from
    Sparse Annotation", MICCAI 2016. https://arxiv.org/abs/1606.06650
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Two consecutive 3×3×3 convolutions each followed by InstanceNorm + LeakyReLU.

    With ``residual=True`` a 1×1×1 projection shortcut is added so the block
    learns a residual mapping, which improves gradient flow at greater depths.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        residual:     Whether to add a skip connection around the two convs.
        dropout:      Dropout3d probability applied between the two convs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # 1×1×1 projection to align channel counts for the residual addition
        self.shortcut: nn.Module = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if residual and in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.residual:
            out = out + self.shortcut(x)
        return out


class EncoderBlock(nn.Module):
    """
    Encoder stage: ConvBlock followed by 2×2×2 max-pooling.

    Returns both the pooled feature map (passed to the next encoder level)
    and the pre-pool activation (stored as a skip connection for the decoder).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, residual=residual, dropout=dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pooled: Downsampled feature map → fed to the next level.
            skip:   Pre-pool activation   → fed to the matching decoder.
        """
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """
    Decoder stage: transposed-conv upsampling, skip concatenation, ConvBlock.

    The transposed conv halves the channel count while doubling the spatial
    resolution; the skip-connection channels are then concatenated before
    the double-conv refinement.

    Args:
        in_channels:   Channels of the feature map from the previous decoder level.
        skip_channels: Channels of the matching encoder skip connection.
        out_channels:  Channels after the ConvBlock refinement.
        residual:      Forward to ConvBlock residual flag.
        dropout:       Forward to ConvBlock dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            residual=residual,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Guard against off-by-one mismatches caused by odd spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    3D U-Net with configurable depth and optional residual connections.

    Feature widths at successive encoder levels:
        [base_features, base_features×2, ..., base_features×2^(depth-1)]
    The bottleneck doubles the deepest encoder width.

    The output is raw logits — apply torch.sigmoid externally for probabilities
    or use BCEWithLogitsLoss during training (numerically more stable).

    Args:
        in_channels:   Input channels (1 for mono-modal CT).
        out_channels:  Output channels (1 for binary liver segmentation).
        base_features: Feature width at encoder level 0.
        depth:         Number of encoder/decoder levels.
        residual:      Enable residual shortcuts in each ConvBlock.
        dropout:       Dropout3d probability (0.0 = disabled).

    Example (default config, depth=4, base_features=32):
        Encoder:    1→32 → 32→64 → 64→128 → 128→256
        Bottleneck: 256→512
        Decoder:    512+256→256 → 256+128→128 → 128+64→64 → 64+32→32
        Output:     32→1
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth

        # Feature width at each encoder level: [32, 64, 128, 256] for depth=4
        features: List[int] = [base_features * (2 ** i) for i in range(depth)]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for out_ch in features:
            self.encoders.append(
                EncoderBlock(in_ch, out_ch, residual=residual, dropout=dropout)
            )
            in_ch = out_ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_ch = features[-1] * 2
        self.bottleneck = ConvBlock(
            features[-1], bottleneck_ch, residual=residual, dropout=dropout
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        in_ch = bottleneck_ch
        for skip_ch in reversed(features):
            self.decoders.append(
                DecoderBlock(in_ch, skip_ch, skip_ch, residual=residual, dropout=dropout)
            )
            in_ch = skip_ch

        # ── Output ───────────────────────────────────────────────────────────
        # 1×1×1 conv to project to output_channels; no activation (raw logits)
        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        self._initialise_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.output_conv(x)

    def _initialise_weights(self) -> None:
        """Kaiming normal init for conv layers; ones/zeros for InstanceNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config: Dict) -> UNet3D:
    """Construct a UNet3D from the 'model' section of config.yaml."""
    cfg = config["model"]
    return UNet3D(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_features=cfg["base_features"],
        depth=cfg["depth"],
        residual=cfg["residual"],
        dropout=cfg.get("dropout", 0.0),
    )
