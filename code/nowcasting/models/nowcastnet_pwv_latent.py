"""Two-stream latent-state fusion for radar/PWV precipitation nowcasting.

PWV is encoded as an observed atmospheric state and fused once at the
generative bottleneck.  It never enters the recursive radar source equation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nowcasting.layers.evolution.evolution_network import Evolution_Network
from nowcasting.layers.generation.generative_network import (
    Generative_Decoder,
    Generative_Encoder,
)
from nowcasting.layers.generation.noise_projector import Noise_Projector
from nowcasting.layers.utils import make_grid, warp
from nowcasting.models.lead_time_conditioning import LeadTimeConditioner


class PWVLatentFusionNet(nn.Module):
    """Fuse independently encoded PWV state into the radar latent feature.

    The radar path has the same modules and tensor names as ``NowcastNet`` so
    it can be initialized from a matched radar checkpoint.  Cross-attention
    uses radar latent tokens as queries and PWV tokens as keys/values.
    """

    def __init__(self, configs):
        super(PWVLatentFusionNet, self).__init__()
        self.configs = configs
        self.pred_length = configs.total_length - configs.input_length
        self.intensity_scale = float(getattr(configs, "intensity_scale", 128.0))
        self.pwv_intensity_scale = float(
            getattr(configs, "pwv_intensity_scale", 80.0)
        )
        evo_base_c = int(getattr(configs, "evo_base_channels", 32))
        pwv_base_c = int(getattr(configs, "pwv_latent_channels", 8))
        radar_dim = 8 * int(configs.ngf)
        pwv_dim = 8 * pwv_base_c
        attention_heads = int(getattr(configs, "pwv_latent_heads", 4))
        if radar_dim % attention_heads:
            raise ValueError(
                "8*ngf={} must be divisible by pwv_latent_heads={}".format(
                    radar_dim, attention_heads
                )
            )

        self.radar_evo_net = Evolution_Network(
            configs.input_length, self.pred_length, base_c=evo_base_c
        )
        self.lead_time = LeadTimeConditioner(
            self.pred_length,
            getattr(configs, "lead_time_embed_dim", 0),
            targets={"source": 1, "motion": 2},
        )
        self.gen_enc = Generative_Encoder(
            configs.total_length, base_c=configs.ngf
        )
        self.gen_dec = Generative_Decoder(configs)
        self.proj = Noise_Projector(configs.ngf, configs)

        self.pwv_encoder = Generative_Encoder(
            configs.input_length, base_c=pwv_base_c
        )
        self.pwv_to_radar = nn.Conv2d(pwv_dim, radar_dim, kernel_size=1)
        self.radar_norm = nn.LayerNorm(radar_dim)
        self.pwv_norm = nn.LayerNorm(radar_dim)
        self.cross_attention = nn.MultiheadAttention(
            radar_dim,
            attention_heads,
            dropout=float(getattr(configs, "pwv_latent_dropout", 0.0)),
            batch_first=True,
        )
        latent_height = int(configs.img_height) // 8
        latent_width = int(configs.img_width) // 8
        self.position_embedding = nn.Parameter(
            torch.zeros(1, latent_height * latent_width, radar_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(radar_dim * 2, radar_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(radar_dim, radar_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.pwv_aux_head = nn.Conv2d(pwv_dim, self.pred_length, kernel_size=1)

        sample = torch.zeros(
            1, 1, configs.img_height, configs.img_width
        )
        self.grid = make_grid(sample)

    def radar_backbone_modules(self):
        return [self.radar_evo_net, self.gen_enc, self.gen_dec, self.proj]

    def fusion_parameters(self):
        modules = [
            self.pwv_encoder,
            self.pwv_to_radar,
            self.radar_norm,
            self.pwv_norm,
            self.cross_attention,
            self.fusion_gate,
            self.pwv_aux_head,
        ]
        yield self.position_embedding
        for module in modules:
            yield from module.parameters()

    def _prepare_pwv(self, pwv_frames, shape, device):
        batch, _, height, width = shape
        if pwv_frames is None:
            return torch.zeros(
                batch,
                self.configs.input_length,
                height,
                width,
                device=device,
            )
        if pwv_frames.dim() == 5:
            pwv_frames = pwv_frames[..., 0]
        return pwv_frames[:, : self.configs.input_length].to(device)

    def _fuse(self, radar_feature, pwv_input):
        pwv_normalized = torch.clamp(
            pwv_input / max(self.pwv_intensity_scale, 1e-6), 0.0, 1.5
        )
        pwv_feature = self.pwv_encoder(pwv_normalized)
        pwv_projected = self.pwv_to_radar(pwv_feature)
        batch, channels, height, width = radar_feature.shape
        radar_tokens = radar_feature.flatten(2).transpose(1, 2)
        pwv_tokens = pwv_projected.flatten(2).transpose(1, 2)
        position = self.position_embedding[:, : radar_tokens.size(1)]
        attended, attention = self.cross_attention(
            self.radar_norm(radar_tokens + position),
            self.pwv_norm(pwv_tokens + position),
            self.pwv_norm(pwv_tokens + position),
            need_weights=True,
            average_attn_weights=True,
        )
        attended_map = attended.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        # Preserve a spatially anchored local path in addition to global
        # cross-attention. This makes geographical alignment testable.
        attended_map = attended_map + pwv_projected
        gate = self.fusion_gate(
            torch.cat([radar_feature, attended_map], dim=1)
        )
        fused = radar_feature + gate * attended_map
        pwv_prediction = torch.sigmoid(self.pwv_aux_head(pwv_feature))
        pwv_prediction = F.interpolate(
            pwv_prediction,
            size=(self.configs.img_height, self.configs.img_width),
            mode="bilinear",
            align_corners=False,
        ) * self.pwv_intensity_scale
        return fused, gate, attention, pwv_prediction

    def forward(self, all_frames, pwv_frames=None, return_aux=False):
        all_frames = all_frames[:, :, :, :, :1]
        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch, _, _, height, width = frames.shape
        input_frames = frames[:, : self.configs.input_length].reshape(
            batch, self.configs.input_length, height, width
        )
        pwv_input = self._prepare_pwv(
            pwv_frames, input_frames.shape, all_frames.device
        )

        intensity, motion = self.radar_evo_net(input_frames)
        intensity = self.lead_time(intensity, "source")
        motion_ = motion.reshape(
            batch, self.pred_length, 2, height, width
        )
        motion_ = self.lead_time(motion_, "motion")
        intensity_ = intensity.reshape(
            batch, self.pred_length, 1, height, width
        )
        last_frames = all_frames[
            :, self.configs.input_length - 1 : self.configs.input_length,
            :, :, 0
        ]
        grid = self.grid.to(all_frames.device).repeat(batch, 1, 1, 1)
        series = []
        advected_series = []
        for index in range(self.pred_length):
            advected = warp(
                last_frames,
                motion_[:, index],
                grid,
                mode="nearest",
                padding_mode="border",
            )
            last_frames = advected + intensity_[:, index]
            advected_series.append(advected)
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)
        advected_result = torch.cat(advected_series, dim=1)
        evo_condition = evo_result / self.intensity_scale

        radar_feature = self.gen_enc(
            torch.cat([input_frames, evo_condition], dim=1)
        )
        fused_feature, gate, attention, pwv_prediction = self._fuse(
            radar_feature, pwv_input
        )
        noise_height, noise_width = height // 32, width // 32
        noise = torch.randn(
            batch,
            self.configs.ngf,
            noise_height,
            noise_width,
            device=all_frames.device,
        )
        noise_feature = (
            self.proj(noise)
            .reshape(batch, -1, 4, 4, noise_height, noise_width)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(batch, -1, height // 8, width // 8)
        )
        feature = torch.cat([fused_feature, noise_feature], dim=1)
        prediction = self.gen_dec(feature, evo_condition).unsqueeze(-1)

        if not return_aux:
            return prediction
        gate_map = F.interpolate(
            gate.mean(dim=1, keepdim=True),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        coupling = gate_map.unsqueeze(1).repeat(
            1, self.pred_length, 1, 1, 1
        )
        return {
            "prediction": prediction,
            "evolution": evo_condition,
            "advected": advected_result,
            "motion": motion_,
            "intensity": intensity_,
            "coupling": coupling,
            "pwv_latent_attention": attention,
            "pwv_prediction": pwv_prediction,
            "pwv_input": pwv_input,
        }
