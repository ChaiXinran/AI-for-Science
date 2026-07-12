import torch
import torch.nn as nn
import torch.nn.functional as F

from nowcasting.layers.evolution.evolution_network import Evolution_Network
from nowcasting.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from nowcasting.layers.generation.noise_projector import Noise_Projector
from nowcasting.layers.utils import make_grid, warp


class ConvBlock(nn.Module):
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
    """Small U-Net used for PWV source and coupling fields.

    Unlike Evolution_Network, the output head is not gated by a zero-initialized
    gamma. This lets the coupling field move away from 0.5 early in training.
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


class PWVCoupledNetV2(nn.Module):
    """PWV-coupled NowcastNet with explicit physical PWV feature modulation.

    The source term is decomposed as:

        s_t = s_t^radar + C_t(x, y) * s_t^pwv

    V2 changes the first PWV version in two ways:
    1. PWV is expanded into value, anomaly, temporal change, and gradient maps.
    2. The coupling field uses a dedicated lightweight U-Net instead of the
       zero-gated Evolution_Network.
    """

    def __init__(self, configs):
        super(PWVCoupledNetV2, self).__init__()
        self.configs = configs
        self.pred_length = configs.total_length - configs.input_length
        evo_base_c = getattr(configs, "evo_base_channels", 32)
        pwv_base_c = getattr(configs, "pwv_base_channels", 24)
        self.intensity_scale = float(getattr(configs, "intensity_scale", 128.0))
        self.pwv_feature_groups = 4
        pwv_channels = configs.input_length * self.pwv_feature_groups
        coupling_channels = configs.input_length + pwv_channels

        self.radar_evo_net = Evolution_Network(configs.input_length, self.pred_length, base_c=evo_base_c)
        self.pwv_source_net = LightweightUNet(pwv_channels, self.pred_length, base_channels=pwv_base_c)
        self.coupling_net = LightweightUNet(coupling_channels, self.pred_length, base_channels=pwv_base_c)

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
        mean = pwv_input.mean(dim=(1, 2, 3), keepdim=True)
        std = pwv_input.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        value = (pwv_input - mean) / std
        anomaly = pwv_input - pwv_input.mean(dim=1, keepdim=True)
        delta = torch.zeros_like(pwv_input)
        delta[:, 1:] = pwv_input[:, 1:] - pwv_input[:, :-1]
        dx = F.pad(pwv_input[..., :, 1:] - pwv_input[..., :, :-1], (0, 1, 0, 0))
        dy = F.pad(pwv_input[..., 1:, :] - pwv_input[..., :-1, :], (0, 0, 0, 1))
        gradient = torch.sqrt(dx * dx + dy * dy + 1e-6)
        features = torch.cat([value, anomaly, delta, gradient], dim=1)
        return torch.clamp(features, -5.0, 5.0)

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

        radar_intensity, motion = self.radar_evo_net(input_frames)
        pwv_intensity = self.pwv_source_net(pwv_features)
        coupling_logits = self.coupling_net(torch.cat([radar_context, pwv_features], dim=1))
        coupling = torch.sigmoid(coupling_logits)

        source = radar_intensity + coupling * pwv_intensity
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        source_ = source.reshape(batch, self.pred_length, 1, height, width)
        radar_source_ = radar_intensity.reshape(batch, self.pred_length, 1, height, width)
        pwv_source_ = pwv_intensity.reshape(batch, self.pred_length, 1, height, width)
        coupling_ = coupling.reshape(batch, self.pred_length, 1, height, width)

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
                "coupling": coupling_,
                "pwv_features": pwv_features,
                "pwv_input": pwv_input,
            }
        return gen_result
