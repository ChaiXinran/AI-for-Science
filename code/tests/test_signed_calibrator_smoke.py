"""Unit tests for the bounded signed PWV calibrator."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nowcasting.experiments.common import load_radar_backbone_weights
from nowcasting.models.registry import build_model
from train.pwv_signed import balanced_threshold_loss


def model_args(device, model_name, climatology_path=""):
    input_length = 2
    total_length = 4
    ngf = 4
    return SimpleNamespace(
        model_name=model_name,
        input_length=input_length,
        total_length=total_length,
        img_height=32,
        img_width=32,
        img_ch=2,
        ngf=ngf,
        evo_ic=total_length - input_length,
        gen_oc=total_length - input_length,
        ic_feature=ngf * 10,
        intensity_scale=35.0,
        pwv_intensity_scale=80.0,
        evo_base_channels=32,
        pwv_base_channels=4,
        fusion_channels=4,
        lead_time_embed_dim=4,
        pwv_source_type="cnn",
        frame_minutes=6.0,
        pwv_tendency_windows="",
        pwv_tendency_mode="slope",
        pwv_candidate_threshold=0.5,
        pwv_candidate_radius=2,
        pwv_climatology_path=str(climatology_path),
        signed_use_tendency=False,
        signed_residual_scale=0.25,
        device=device,
    )


class SignedCalibratorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_null_identity_signed_bound_and_gradients(self):
        torch.manual_seed(2026)
        with tempfile.TemporaryDirectory() as temp:
            climatology = Path(temp) / "climatology.npz"
            np.savez_compressed(
                climatology,
                mean=np.full((20, 24), 30.0, dtype=np.float32),
                std=np.full((20, 24), 5.0, dtype=np.float32),
            )
            radar_args = model_args(self.device, "NowcastNet")
            signed_args = model_args(
                self.device, "PWVSignedCalibratorNowcastNet", climatology
            )
            radar_model = build_model(radar_args).to(self.device).eval()
            signed_model = build_model(signed_args).to(self.device).eval()
            checkpoint = Path(temp) / "radar.ckpt"
            torch.save(radar_model.state_dict(), checkpoint)
            load_radar_backbone_weights(signed_model, checkpoint, self.device)

            frames = torch.rand(1, 4, 32, 32, 2, device=self.device) * 20.0
            null_pwv = torch.zeros(1, 4, 32, 32, device=self.device)
            real_pwv = torch.rand_like(null_pwv) * 80.0
            with torch.no_grad():
                torch.manual_seed(99)
                radar_prediction = radar_model(frames)
                torch.manual_seed(99)
                null_aux = signed_model(frames, null_pwv, return_aux=True)
                torch.manual_seed(99)
                real_aux = signed_model(frames, real_pwv, return_aux=True)

            self.assertTrue(
                torch.equal(
                    null_aux["pwv_contribution"],
                    torch.zeros_like(null_aux["pwv_contribution"]),
                )
            )
            self.assertTrue(
                torch.allclose(
                    null_aux["prediction"], radar_prediction, atol=1e-6, rtol=1e-6
                )
            )
            self.assertTrue(torch.isfinite(real_aux["prediction"]).all().item())
            self.assertLessEqual(
                float(real_aux["pwv_contribution"].abs().max()),
                signed_args.signed_residual_scale + 1e-6,
            )

            signed_model.freeze_radar_backbone()
            signed_model.train()
            aux = signed_model(frames, real_pwv, return_aux=True)
            target = frames[:, 2:, :, :, 0]
            threshold_loss, _ = balanced_threshold_loss(
                aux["prediction"][..., 0], target, [10.0, 20.0], 1.0
            )
            threshold_loss.backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in signed_model.signed_condition_net.parameters()
                )
            )
            self.assertTrue(
                all(
                    parameter.grad is None
                    for parameter in signed_model.radar_evo_net.parameters()
                )
            )


if __name__ == "__main__":
    unittest.main()
