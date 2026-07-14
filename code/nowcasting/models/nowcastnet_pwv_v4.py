import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nowcasting.layers.evolution.evolution_network import Evolution_Network
from nowcasting.layers.generation.generative_network import Generative_Decoder, Generative_Encoder
from nowcasting.layers.generation.noise_projector import Noise_Projector
from nowcasting.layers.utils import make_grid, warp
from nowcasting.models.lead_time_conditioning import LeadTimeConditioner
from nowcasting.models.nowcastnet_pwv_v2 import ConvBlock, LightweightUNet
from nowcasting.models.pwv_features import base_pwv_features, build_pwv_features, pwv_feature_group_count


class TemporalPWVCrossAttentionSource(nn.Module):
    """Radar-query/PWV-key-value source generator.

    Attention is computed independently for each low-resolution spatial
    location over the PWV history. This keeps the first version cheap while
    letting the radar state choose which PWV lead-in times are useful.
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

    def _pwv_sequence_features(self, pwv_input):
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


class PWVCoupledNetV4(nn.Module):
    """PWV-gated NowcastNet with radar-query temporal PWV cross-attention."""

    def __init__(self, configs):
        super(PWVCoupledNetV4, self).__init__()
        self.configs = configs
        self.pred_length = configs.total_length - configs.input_length
        evo_base_c = getattr(configs, "evo_base_channels", 32)
        pwv_base_c = getattr(configs, "pwv_base_channels", 24)
        fusion_channels = getattr(configs, "fusion_channels", 32)
        attn_dim = getattr(configs, "pwv_attn_dim", 64)
        attn_heads = getattr(configs, "pwv_attn_heads", 4)
        attn_downsample = getattr(configs, "pwv_attn_downsample", 4)
        attn_source_scale = getattr(configs, "pwv_attn_source_scale", 0.0)
        self.intensity_scale = float(getattr(configs, "intensity_scale", 128.0))
        self.pwv_feature_groups = pwv_feature_group_count(configs)
        pwv_channels = configs.input_length * self.pwv_feature_groups

        self.radar_evo_net = Evolution_Network(configs.input_length, self.pred_length, base_c=evo_base_c)
        self.radar_stem = ConvBlock(configs.input_length, fusion_channels)
        self.pwv_stem = ConvBlock(pwv_channels, fusion_channels)
        self.feature_gate = nn.Sequential(
            ConvBlock(fusion_channels * 2, fusion_channels),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=1),
        )

        self.pwv_source_net = TemporalPWVCrossAttentionSource(
            configs.input_length,
            self.pred_length,
            configs.input_length,
            fusion_channels,
            attn_dim=attn_dim,
            heads=attn_heads,
            downsample=attn_downsample,
            source_scale=attn_source_scale,
            intensity_scale=self.intensity_scale,
        )
        self.source_coupling_net = LightweightUNet(
            configs.input_length + pwv_channels + fusion_channels,
            self.pred_length,
            base_channels=pwv_base_c,
        )
        self.support_gate_net = LightweightUNet(
            configs.input_length + pwv_channels + fusion_channels,
            self.pred_length,
            base_channels=pwv_base_c,
            final_bias=-1.0,
        )
        self.lead_time = LeadTimeConditioner(
            self.pred_length,
            getattr(configs, "lead_time_embed_dim", 0),
        )

        self.gen_enc = Generative_Encoder(configs.total_length, base_c=configs.ngf)
        self.gen_dec = Generative_Decoder(configs)
        self.proj = Noise_Projector(configs.ngf, configs)

        sample_tensor = torch.zeros(1, 1, configs.img_height, configs.img_width)
        self.grid = make_grid(sample_tensor)

    def _prepare_pwv_input(self, input_frames, pwv_frames):
        if pwv_frames is None:
            return torch.zeros_like(input_frames)
        if pwv_frames.dim() == 5:
            pwv_frames = pwv_frames[..., 0]
        return pwv_frames[:, :self.configs.input_length].to(input_frames.device)

    def _pwv_features(self, pwv_input):
        return build_pwv_features(pwv_input, self.configs)

    def forward(self, all_frames, pwv_frames=None, return_aux=False):
        all_frames = all_frames[:, :, :, :, :1]
        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        input_frames = frames[:, :self.configs.input_length]
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width)
        pwv_input = self._prepare_pwv_input(input_frames, pwv_frames)
        pwv_features = self._pwv_features(pwv_input)
        radar_context = torch.clamp(input_frames / max(self.intensity_scale, 1.0), 0.0, 1.0)

        radar_feat = self.radar_stem(radar_context)
        pwv_feat = self.pwv_stem(pwv_features)
        feature_coupling = torch.sigmoid(self.feature_gate(torch.cat([radar_feat, pwv_feat], dim=1)))
        fused_feature = radar_feat + feature_coupling * pwv_feat

        radar_intensity, motion = self.radar_evo_net(input_frames)
        gate_input = torch.cat([radar_context, pwv_features, fused_feature], dim=1)
        pwv_intensity, pwv_attention = self.pwv_source_net(radar_context, pwv_input, fused_feature)
        source_coupling_logits = self.lead_time(self.source_coupling_net(gate_input), "gate")
        support_gate_logits = self.lead_time(self.support_gate_net(gate_input), "gate")
        source_coupling = torch.sigmoid(source_coupling_logits)
        support_gate = torch.sigmoid(support_gate_logits)

        pwv_contribution = source_coupling * support_gate * pwv_intensity
        source = radar_intensity + pwv_contribution
        source = self.lead_time(source, "source")
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        motion_ = self.lead_time(motion_, "motion")
        source_ = source.reshape(batch, self.pred_length, 1, height, width)
        radar_source_ = radar_intensity.reshape(batch, self.pred_length, 1, height, width)
        pwv_source_ = pwv_intensity.reshape(batch, self.pred_length, 1, height, width)
        source_coupling_ = source_coupling.reshape(batch, self.pred_length, 1, height, width)
        support_gate_ = support_gate.reshape(batch, self.pred_length, 1, height, width)

        series = []
        advected_series = []
        last_frames = all_frames[:, (self.configs.input_length - 1):self.configs.input_length, :, :, 0]
        grid = self.grid.to(all_frames.device).repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):
            advected = warp(last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border")
            last_frames = advected + source_[:, i]
            advected_series.append(advected)
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)
        advected_result = torch.cat(advected_series, dim=1)
        evo_condition = evo_result / self.intensity_scale

        evo_feature = self.gen_enc(torch.cat([input_frames, evo_condition], dim=1))
        noise = torch.randn(batch, self.configs.ngf, height // 32, width // 32, device=all_frames.device)
        noise_feature = (
            self.proj(noise)
            .reshape(batch, -1, 4, 4, 8, 8)
            .permute(0, 1, 4, 5, 2, 3)
            .reshape(batch, -1, height // 8, width // 8)
        )

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_condition).unsqueeze(-1)

        if return_aux:
            return {
                "prediction": gen_result,
                "evolution": evo_condition,
                "advected": advected_result,
                "motion": motion_,
                "intensity": source_,
                "radar_source": radar_source_,
                "pwv_source": pwv_source_,
                "pwv_contribution": pwv_contribution.reshape(batch, self.pred_length, 1, height, width),
                "coupling": source_coupling_,
                "support_gate": support_gate_,
                "feature_coupling": feature_coupling,
                "pwv_temporal_attention": pwv_attention,
                "pwv_features": pwv_features,
                "pwv_input": pwv_input,
            }
        return gen_result
