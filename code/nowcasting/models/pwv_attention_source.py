import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nowcasting.layers.shared_blocks import ConvBlock, LightweightUNet
from nowcasting.models.pwv_features import base_pwv_features


class TemporalPWVCrossAttentionSource(nn.Module):
    """Pluggable PWV source generator with radar-query temporal cross-attention.

    Each low-resolution spatial location independently queries the 9-frame
    PWV history.  This lets the radar state choose which PWV lead-in times
    are useful at each position — fast-developing convection may attend to
    the most recent frames while large-scale stratiform systems may spread
    attention across the full history.

    Usage::

        # CNN mode (LightweightUNet, V3 default)
        net = LightweightUNet(pwv_channels + fusion_channels, pred_length)

        # Attention mode (this module)
        net = TemporalPWVCrossAttentionSource(
            input_length, pred_length, radar_channels, fusion_channels, ...
        )

    Select via ``--pwv_source_type cnn|attention`` in training / test scripts.
    """

    def __init__(
        self,
        input_length,
        pred_length,
        radar_channels,
        fusion_channels,
        attn_dim=64,
        heads=4,
        downsample=4,
        source_scale=0.0,
        intensity_scale=128.0,
    ):
        super(TemporalPWVCrossAttentionSource, self).__init__()
        self.input_length = int(input_length)
        self.pred_length = int(pred_length)
        self.attn_dim = int(attn_dim)
        self.heads = max(1, int(heads))
        if self.attn_dim % self.heads != 0:
            raise ValueError("pwv_attn_dim must be divisible by pwv_attn_heads.")
        self.head_dim = self.attn_dim // self.heads
        self.downsample = max(1, int(downsample))
        self.source_scale = float(source_scale)
        if self.source_scale <= 0:
            self.source_scale = 0.35 * float(intensity_scale)

        self.radar_query_encoder = ConvBlock(radar_channels + fusion_channels, self.attn_dim)
        self.pwv_key_encoder = ConvBlock(4, self.attn_dim)
        self.q_proj = nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=1)
        self.source_decoder = LightweightUNet(
            self.attn_dim + fusion_channels,
            self.pred_length,
            base_channels=max(8, self.attn_dim // 2),
        )

    def _pool(self, x):
        if self.downsample <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.downsample, stride=self.downsample)

    @staticmethod
    def _pwv_sequence_features(pwv_input):
        features = torch.stack(base_pwv_features(pwv_input), dim=2)
        return torch.clamp(features, -5.0, 5.0)

    def forward(self, radar_context, pwv_input, fused_feature):
        batch, _, height, width = radar_context.shape
        radar_low = self._pool(radar_context)
        fused_low = self._pool(fused_feature)
        low_h, low_w = radar_low.shape[-2:]

        query_input = torch.cat([radar_low, fused_low], dim=1)
        query_feature = self.radar_query_encoder(query_input)
        query = self.q_proj(query_feature)

        pwv_seq = self._pwv_sequence_features(pwv_input)
        pwv_seq = pwv_seq.reshape(batch * self.input_length, 4, height, width)
        pwv_seq = self._pool(pwv_seq)
        pwv_feature = self.pwv_key_encoder(pwv_seq)
        key = self.k_proj(pwv_feature).reshape(batch, self.input_length, self.attn_dim, low_h, low_w)
        value = self.v_proj(pwv_feature).reshape(batch, self.input_length, self.attn_dim, low_h, low_w)

        num_points = low_h * low_w
        query = query.flatten(2).transpose(1, 2)
        key = key.flatten(3).permute(0, 3, 1, 2)
        value = value.flatten(3).permute(0, 3, 1, 2)

        query = query.reshape(batch, num_points, self.heads, self.head_dim)
        key = key.reshape(batch, num_points, self.input_length, self.heads, self.head_dim)
        value = value.reshape(batch, num_points, self.input_length, self.heads, self.head_dim)

        logits = (query.unsqueeze(2) * key).sum(dim=-1) / math.sqrt(self.head_dim)
        weights = torch.softmax(logits, dim=2)
        context = (weights.unsqueeze(-1) * value).sum(dim=2)
        context = context.reshape(batch, num_points, self.attn_dim).transpose(1, 2)
        context = context.reshape(batch, self.attn_dim, low_h, low_w)

        source_low = self.source_decoder(torch.cat([context, fused_low], dim=1))
        source_low = torch.tanh(source_low) * self.source_scale
        source = F.interpolate(source_low, size=(height, width), mode="bilinear", align_corners=False)

        attention = weights.mean(dim=-1).permute(0, 2, 1).reshape(batch, self.input_length, low_h, low_w)
        return source, attention
