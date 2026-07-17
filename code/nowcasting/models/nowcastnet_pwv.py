import torch
import torch.nn as nn
import torch.nn.functional as F

from nowcasting.layers.evolution.evolution_network import Evolution_Network
from nowcasting.layers.generation.generative_network import Generative_Decoder, Generative_Encoder
from nowcasting.layers.generation.noise_projector import Noise_Projector
from nowcasting.layers.utils import make_grid, warp
from nowcasting.models.lead_time_conditioning import LeadTimeConditioner
from nowcasting.layers.shared_blocks import ConvBlock, LightweightUNet
from nowcasting.models.pwv_attention_source import TemporalPWVCrossAttentionSource
from nowcasting.models.pwv_features import build_base_pwv_features, build_pwv_features, pwv_feature_group_count


class PWVCoupledNet(nn.Module):
    """PWV-gated NowcastNet with false-alarm control and optional attention source.

    V2 used PWV as an additive source modulation:

        s = s_radar + C_s * s_pwv

    V3 keeps that physical source decomposition, but also learns a feature-space
    support gate:

        F_fuse = Z_r + C_f * A_pwv
        s = s_radar + C_s * S_pwv * s_pwv

    `S_pwv` is intended to suppress PWV contributions where the moisture field
    does not support precipitation maintenance or growth.
    """

    def __init__(self, configs):
        super(PWVCoupledNet, self).__init__()
        self.configs = configs
        self.pred_length = configs.total_length - configs.input_length
        evo_base_c = getattr(configs, "evo_base_channels", 32)
        pwv_base_c = getattr(configs, "pwv_base_channels", 24)
        fusion_channels = getattr(configs, "fusion_channels", 32)
        self.intensity_scale = float(getattr(configs, "intensity_scale", 128.0))
        self.pwv_source_type = getattr(configs, "pwv_source_type", "cnn")
        self.pwv_feature_groups = pwv_feature_group_count(configs)
        self.has_pwv_tendency = self.pwv_feature_groups > 4
        base_pwv_channels = configs.input_length * 4
        pwv_channels = configs.input_length * self.pwv_feature_groups

        self.radar_evo_net = Evolution_Network(configs.input_length, self.pred_length, base_c=evo_base_c)
        self.radar_stem = ConvBlock(configs.input_length, fusion_channels)
        self.pwv_stem = ConvBlock(pwv_channels, fusion_channels)
        self.feature_gate = nn.Sequential(
            ConvBlock(fusion_channels * 2, fusion_channels),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=1),
        )

        source_pwv_channels = base_pwv_channels if self.has_pwv_tendency else pwv_channels
        if self.has_pwv_tendency:
            self.base_pwv_stem = ConvBlock(base_pwv_channels, fusion_channels)
            self.base_feature_gate = nn.Sequential(
                ConvBlock(fusion_channels * 2, fusion_channels),
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=1),
            )
        if self.pwv_source_type == "attention":
            self.pwv_source_net = TemporalPWVCrossAttentionSource(
                configs.input_length,
                self.pred_length,
                configs.input_length,
                fusion_channels,
                attn_dim=getattr(configs, "pwv_attn_dim", 64),
                heads=getattr(configs, "pwv_attn_heads", 4),
                downsample=getattr(configs, "pwv_attn_downsample", 4),
                source_scale=getattr(configs, "pwv_attn_source_scale", 0.0),
                intensity_scale=self.intensity_scale,
            )
        else:
            self.pwv_source_net = LightweightUNet(
                source_pwv_channels + fusion_channels,
                self.pred_length,
                base_channels=pwv_base_c,
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

    def _base_pwv_features(self, pwv_input):
        return build_base_pwv_features(pwv_input)

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
        base_pwv_features = self._base_pwv_features(pwv_input)
        radar_context = torch.clamp(input_frames / max(self.intensity_scale, 1.0), 0.0, 1.0)

        radar_feat = self.radar_stem(radar_context)
        pwv_feat = self.pwv_stem(pwv_features)
        feature_coupling = torch.sigmoid(self.feature_gate(torch.cat([radar_feat, pwv_feat], dim=1)))
        fused_feature = radar_feat + feature_coupling * pwv_feat
        source_pwv_features = pwv_features
        source_fused_feature = fused_feature
        if self.has_pwv_tendency:
            base_pwv_feat = self.base_pwv_stem(base_pwv_features)
            base_feature_coupling = torch.sigmoid(self.base_feature_gate(torch.cat([radar_feat, base_pwv_feat], dim=1)))
            source_pwv_features = base_pwv_features
            source_fused_feature = radar_feat + base_feature_coupling * base_pwv_feat

        radar_intensity, motion = self.radar_evo_net(input_frames)
        gate_input = torch.cat([radar_context, pwv_features, fused_feature], dim=1)
        pwv_attention = None
        if self.pwv_source_type == "attention":
            pwv_intensity, pwv_attention = self.pwv_source_net(
                radar_context, pwv_input, fused_feature
            )
        else:
            pwv_intensity = self.pwv_source_net(
                torch.cat([source_pwv_features, source_fused_feature], dim=1)
            )
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
            aux = {
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
                "pwv_features": pwv_features,
                "pwv_input": pwv_input,
            }
            if pwv_attention is not None:
                aux["pwv_temporal_attention"] = pwv_attention
            return aux
        return gen_result
