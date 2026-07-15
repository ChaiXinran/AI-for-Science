import torch
import torch.nn as nn

from nowcasting.models.nowcastnet_pwv_v2 import LightweightUNet


class ObjectPredictionHead(nn.Module):
    """Auxiliary object head for future convective-object supervision."""

    def __init__(self, pred_length, base_channels=24, intensity_scale=128.0):
        super(ObjectPredictionHead, self).__init__()
        self.pred_length = pred_length
        self.intensity_scale = float(intensity_scale)
        self.in_channels_per_lead = 6
        self.out_channels_per_lead = 5
        self.net = LightweightUNet(
            pred_length * self.in_channels_per_lead,
            pred_length * self.out_channels_per_lead,
            base_channels=base_channels,
        )

    def _as_lead_map(self, tensor):
        if tensor.dim() == 5 and tensor.size(2) == 1:
            return tensor[:, :, 0]
        return tensor

    def forward(self, aux):
        prediction = aux["prediction"][..., 0]
        evolution = self._as_lead_map(aux["evolution"])
        intensity = self._as_lead_map(aux["intensity"])
        coupling = self._as_lead_map(aux.get("coupling", torch.zeros_like(intensity)))
        support = self._as_lead_map(aux.get("support_gate", torch.ones_like(intensity)))
        motion = aux["motion"]
        motion_magnitude = torch.sqrt((motion * motion).sum(dim=2) + 1e-6)

        scale = max(self.intensity_scale, 1.0)
        context = torch.stack(
            [
                torch.clamp(prediction / scale, 0.0, 1.0),
                torch.clamp(evolution, 0.0, 1.0),
                torch.clamp(intensity / scale, -1.0, 1.0),
                torch.clamp(coupling, 0.0, 1.0),
                torch.clamp(support, 0.0, 1.0),
                torch.clamp(motion_magnitude / 12.0, 0.0, 1.0),
            ],
            dim=2,
        )
        batch, leads, channels, height, width = context.shape
        context = context.reshape(batch, leads * channels, height, width)
        raw = self.net(context).reshape(
            batch,
            self.pred_length,
            self.out_channels_per_lead,
            height,
            width,
        )
        return {
            "raw": raw,
            "center_logits": raw[:, :, 0],
            "mask_logits": raw[:, :, 1],
            "area": torch.sigmoid(raw[:, :, 2]),
            "mean_intensity": torch.sigmoid(raw[:, :, 3]),
            "max_intensity": torch.sigmoid(raw[:, :, 4]),
        }
