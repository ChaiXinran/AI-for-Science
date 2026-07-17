"""Shared building blocks used across NowcastNet model variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two-convolution residual block with GroupNorm + SiLU."""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LightweightUNet(nn.Module):
    """Small U-Net for PWV source and coupling field prediction.

    Three-level encoder-decoder with skip connections.  Unlike
    ``Evolution_Network`` the output head is not gated by a zero-initialized
    gamma, which lets fields move away from zero early in training.
    """

    def __init__(self, in_channels, out_channels, base_channels=24, final_bias=0.0):
        super(LightweightUNet, self).__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.mid = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out.bias, final_bias)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(F.avg_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.enc3(F.avg_pool2d(x2, kernel_size=2, stride=2))
        xm = self.mid(x3)
        y2 = F.interpolate(xm, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        return self.out(y1)
