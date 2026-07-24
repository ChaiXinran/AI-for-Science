import torch

from nowcasting.models.survival_intensity_adapter import (
    DenseSurvivalIntensityAdapter,
    causal_pwv_state,
)


def test_adapter_initializes_as_exact_radar_identity():
    adapter = DenseSurvivalIntensityAdapter(hidden_channels=8)
    observed = torch.rand(2, 9, 16, 16)
    forecast = torch.rand(2, 20, 16, 16)
    auxiliary = torch.rand(2, 6, 16, 16)
    result = adapter(observed, forecast, auxiliary)
    torch.testing.assert_close(result["prediction"], forecast)
    torch.testing.assert_close(result["correction"], torch.zeros_like(forecast))


def test_disabled_adapter_is_identity_after_parameter_change():
    adapter = DenseSurvivalIntensityAdapter(hidden_channels=8)
    torch.nn.init.constant_(adapter.amount_head.bias, 1.0)
    observed = torch.ones(1, 9, 8, 8)
    forecast = torch.ones(1, 20, 8, 8)
    result = adapter(observed, forecast, enabled=False)
    torch.testing.assert_close(result["prediction"], forecast)


def test_causal_pwv_state_has_six_finite_channels():
    history = torch.linspace(20.0, 26.0, 7).view(1, 7, 1, 1).expand(2, 7, 8, 8)
    climatology = torch.full((1, 1, 8, 8), 22.0)
    features = causal_pwv_state(history, climatology)
    assert features.shape == (2, 6, 8, 8)
    assert torch.isfinite(features).all()


def test_candidate_support_is_defined_only_by_radar():
    adapter = DenseSurvivalIntensityAdapter(
        hidden_channels=8, candidate_threshold_mm_per_h=0.5, candidate_radius=0
    )
    observed = torch.zeros(1, 9, 8, 8)
    forecast = torch.zeros(1, 20, 8, 8)
    auxiliary = torch.ones(1, 6, 8, 8)
    torch.nn.init.constant_(adapter.amount_head.bias, 1.0)
    result = adapter(observed, forecast, auxiliary)
    torch.testing.assert_close(result["prediction"], forecast)
