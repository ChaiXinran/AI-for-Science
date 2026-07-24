"""Output-space adapter for echo survival and intensity calibration.

The adapter never predicts a precipitation field from PWV alone.  It receives
the frozen radar forecast, observed-radar summaries, and optional causal PWV
state.  Its correction is bounded and restricted to support proposed by radar,
so disabling the adapter exactly recovers the radar-only forecast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def radar_state_features(observed_radar):
    """Return dense radar-state channels from [B, T, H, W] history."""
    recent = observed_radar[:, -1:]
    early = observed_radar[:, : max(1, observed_radar.shape[1] // 3)].mean(
        dim=1, keepdim=True
    )
    mean = observed_radar.mean(dim=1, keepdim=True)
    maximum = observed_radar.amax(dim=1, keepdim=True)
    tendency = recent - early
    return torch.cat([recent, mean, maximum, tendency], dim=1)


def causal_pwv_state(history, climatology, scale=80.0):
    """Encode causal native-cadence anchors without creating interpolated frames.

    ``history`` is [B, A, H, W] and ``climatology`` is broadcastable to it.
    The six output channels separate static geography from event departures.
    """
    history = history.float()
    climatology = climatology.float()
    if climatology.ndim == 3:
        climatology = climatology.unsqueeze(0)
    if climatology.shape[1] != 1:
        climatology = climatology.mean(dim=1, keepdim=True)
    valid = (climatology > 1e-6).to(history.dtype)
    residual = (history - climatology) * valid
    valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    scalar = residual.sum(dim=(-2, -1), keepdim=True) / valid_count
    anomaly = (residual - scalar * valid) * valid
    scalar_last = scalar[:, -1] * valid[:, 0]
    scalar_mean = scalar.mean(dim=1) * valid[:, 0]
    scalar_tendency = (scalar[:, -1] - scalar[:, 0]) * valid[:, 0]
    anomaly_last = anomaly[:, -1]
    anomaly_tendency = anomaly[:, -1] - anomaly[:, 0]
    static = climatology[:, 0].expand(history.shape[0], -1, -1)
    features = torch.stack(
        [
            static,
            scalar_last,
            scalar_mean,
            scalar_tendency,
            anomaly_last,
            anomaly_tendency,
        ],
        dim=1,
    )
    return (features / max(float(scale), 1e-6)).clamp(-2.0, 2.0)


class DenseSurvivalIntensityAdapter(nn.Module):
    """Shared-lead bounded residual adapter with a radar-defined support mask."""

    auxiliary_channels = 6
    radar_channels = 4

    def __init__(
        self,
        hidden_channels=32,
        max_correction_mm_per_h=12.0,
        candidate_threshold_mm_per_h=0.5,
        candidate_radius=3,
    ):
        super().__init__()
        input_channels = (
            self.radar_channels
            + self.auxiliary_channels
            + 2  # current lead forecast and normalized lead time
        )
        self.max_correction = float(max_correction_mm_per_h)
        self.candidate_threshold = float(candidate_threshold_mm_per_h)
        self.candidate_radius = int(candidate_radius)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.gate_head = nn.Conv2d(hidden_channels, 1, 1)
        self.amount_head = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.constant_(self.gate_head.bias, -1.5)
        nn.init.zeros_(self.amount_head.weight)
        nn.init.zeros_(self.amount_head.bias)

    def candidate_mask(self, observed_radar, radar_forecast):
        recent_support = observed_radar.amax(dim=1, keepdim=True)
        support = torch.maximum(recent_support, radar_forecast.amax(dim=1, keepdim=True))
        support = (support >= self.candidate_threshold).to(radar_forecast.dtype)
        if self.candidate_radius > 0:
            kernel = 2 * self.candidate_radius + 1
            support = F.max_pool2d(
                support, kernel_size=kernel, stride=1, padding=self.candidate_radius
            )
        return support

    def forward(self, observed_radar, radar_forecast, auxiliary=None, enabled=True):
        """Return corrected [B,T,H,W] rain and interpretable adapter fields."""
        if auxiliary is None:
            auxiliary = radar_forecast.new_zeros(
                radar_forecast.shape[0],
                self.auxiliary_channels,
                radar_forecast.shape[-2],
                radar_forecast.shape[-1],
            )
        if auxiliary.shape[1] != self.auxiliary_channels:
            raise ValueError(
                "expected {} auxiliary channels, got {}".format(
                    self.auxiliary_channels, auxiliary.shape[1]
                )
            )
        return self.forward_from_state(
            radar_state_features(observed_radar),
            radar_forecast,
            auxiliary,
            candidate=self.candidate_mask(observed_radar, radar_forecast),
            enabled=enabled,
        )

    def forward_from_state(
        self, radar_state, radar_forecast, auxiliary, candidate, enabled=True
    ):
        """Vectorized adapter path for cached radar summaries."""
        if auxiliary is None:
            auxiliary = radar_forecast.new_zeros(
                radar_forecast.shape[0],
                self.auxiliary_channels,
                radar_forecast.shape[-2],
                radar_forecast.shape[-1],
            )
        if auxiliary.shape[1] != self.auxiliary_channels:
            raise ValueError(
                "expected {} auxiliary channels, got {}".format(
                    self.auxiliary_channels, auxiliary.shape[1]
                )
            )
        batch, lead_count, height, width = radar_forecast.shape
        radar_by_lead = (
            radar_state[:, None]
            .expand(-1, lead_count, -1, -1, -1)
            .reshape(batch * lead_count, self.radar_channels, height, width)
        )
        auxiliary_by_lead = (
            auxiliary[:, None]
            .expand(-1, lead_count, -1, -1, -1)
            .reshape(batch * lead_count, self.auxiliary_channels, height, width)
        )
        forecast_by_lead = radar_forecast.reshape(
            batch * lead_count, 1, height, width
        )
        lead_values = torch.linspace(
            1.0 / lead_count,
            1.0,
            lead_count,
            device=radar_forecast.device,
            dtype=radar_forecast.dtype,
        )
        lead_map = (
            lead_values.view(1, lead_count, 1, 1, 1)
            .expand(batch, -1, -1, height, width)
            .reshape(batch * lead_count, 1, height, width)
        )
        hidden = self.encoder(
            torch.cat(
                [radar_by_lead, auxiliary_by_lead, forecast_by_lead, lead_map],
                dim=1,
            )
        )
        gate = torch.sigmoid(self.gate_head(hidden)).reshape(
            batch, lead_count, height, width
        )
        amount = (
            torch.tanh(self.amount_head(hidden)) * self.max_correction
        ).reshape(batch, lead_count, height, width)
        correction = candidate * gate * amount
        if not enabled:
            correction = torch.zeros_like(correction)
        corrected = (radar_forecast + correction).clamp_min(0.0)
        return {
            "prediction": corrected,
            "radar_prediction": radar_forecast,
            "correction": correction,
            "gate": gate,
            "amount": amount,
            "candidate_mask": candidate,
        }
