"""Smoke tests for end-to-end PWV latent-state fusion."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from nowcasting.experiments.common import load_radar_backbone_weights
from nowcasting.models.registry import build_model
from train.pwv_latent import controlled_training_batch


def model_args(model_name):
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
        lead_time_embed_dim=4,
        pwv_latent_channels=2,
        pwv_latent_heads=4,
        pwv_latent_dropout=0.0,
    )


class LatentFusionSmokeTest(unittest.TestCase):
    def test_shapes_no_future_leakage_and_gradients(self):
        torch.manual_seed(2026)
        radar = build_model(model_args("NowcastNet")).eval()
        fusion = build_model(model_args("PWVLatentFusionNowcastNet")).eval()
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "radar.ckpt"
            torch.save(radar.state_dict(), checkpoint)
            report = load_radar_backbone_weights(fusion, checkpoint, "cpu")
        self.assertGreater(report["loaded_tensors"], 0)

        frames = torch.rand(1, 4, 32, 32, 2) * 20.0
        pwv_a = torch.rand(1, 4, 32, 32) * 80.0
        pwv_b = pwv_a.clone()
        pwv_b[:, 2:] = torch.rand_like(pwv_b[:, 2:]) * 80.0
        with torch.no_grad():
            torch.manual_seed(99)
            aux_a = fusion(frames, pwv_a, return_aux=True)
            torch.manual_seed(99)
            aux_b = fusion(frames, pwv_b, return_aux=True)
        self.assertEqual(aux_a["prediction"].shape, (1, 2, 32, 32, 1))
        self.assertEqual(aux_a["pwv_prediction"].shape, (1, 2, 32, 32))
        self.assertEqual(aux_a["coupling"].shape, (1, 2, 1, 32, 32))
        self.assertTrue(torch.equal(aux_a["prediction"], aux_b["prediction"]))

        fusion.train()
        torch.manual_seed(7)
        aux = fusion(frames, pwv_a, return_aux=True)
        loss = aux["prediction"].abs().mean() + aux["pwv_prediction"].mean()
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in fusion.pwv_encoder.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in fusion.gen_dec.parameters()
            )
        )

    def test_spatial_auxiliary_target_matches_shifted_sequence(self):
        pwv = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(
            1, 4, 8, 8
        )
        torch.manual_seed(2026)
        controlled, target = controlled_training_batch(
            pwv, "spatial_shift", input_length=2
        )
        self.assertTrue(torch.equal(target, controlled[:, 2:]))
        self.assertFalse(torch.equal(target, pwv[:, 2:]))


if __name__ == "__main__":
    unittest.main()
