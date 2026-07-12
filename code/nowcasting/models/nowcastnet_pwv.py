import torch
import torch.nn as nn

from nowcasting.layers.evolution.evolution_network import Evolution_Network
from nowcasting.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from nowcasting.layers.generation.noise_projector import Noise_Projector
from nowcasting.layers.utils import make_grid, warp


class PWVCoupledNet(nn.Module):
    """NowcastNet variant with an explicit PWV coupling source term.

    The evolution source is decomposed as:

        s_t = s_t^radar + C_t(x, y) * s_t^pwv

    where C_t is a sigmoid-bounded coupling field that can be visualized.
    The generative decoder and evolution operator remain compatible with the
    original NowcastNet structure.
    """

    def __init__(self, configs):
        super(PWVCoupledNet, self).__init__()
        self.configs = configs
        self.pred_length = configs.total_length - configs.input_length
        base_c = getattr(configs, "evo_base_channels", 32)

        self.radar_evo_net = Evolution_Network(configs.input_length, self.pred_length, base_c=base_c)
        self.pwv_source_net = Evolution_Network(configs.input_length, self.pred_length, base_c=base_c)
        self.coupling_net = Evolution_Network(configs.input_length * 2, self.pred_length, base_c=base_c)

        self.gen_enc = Generative_Encoder(configs.total_length, base_c=configs.ngf)
        self.gen_dec = Generative_Decoder(configs)
        self.proj = Noise_Projector(configs.ngf, configs)

        sample_tensor = torch.zeros(1, 1, configs.img_height, configs.img_width)
        self.grid = make_grid(sample_tensor)

    def forward(self, all_frames, pwv_frames=None, return_aux=False):
        all_frames = all_frames[:, :, :, :, :1]
        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        input_frames = frames[:, :self.configs.input_length]
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width)

        if pwv_frames is None:
            pwv_input = torch.zeros_like(input_frames)
        else:
            if pwv_frames.dim() == 5:
                pwv_frames = pwv_frames[..., 0]
            pwv_input = pwv_frames[:, :self.configs.input_length].to(all_frames.device)

        radar_intensity, motion = self.radar_evo_net(input_frames)
        pwv_intensity, _ = self.pwv_source_net(pwv_input)
        coupling_logits, _ = self.coupling_net(torch.cat([input_frames, pwv_input], dim=1))
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
        evo_condition = evo_result / 128.0

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
            }
        return gen_result
