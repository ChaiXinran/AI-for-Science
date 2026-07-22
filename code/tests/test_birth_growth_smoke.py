"""Small, self-contained smoke tests for the frozen PWV birth/growth protocol.

Run from the repository root with a CUDA-enabled Python when available:

    python -m unittest code.tests.test_birth_growth_smoke -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from nowcasting.birth_growth import (
    BirthGrowthAccumulator,
    apply_pwv_control,
    birth_growth_losses,
)
from nowcasting.data_provider.custom_png import PngSequenceDataset
from nowcasting.experiments.common import load_radar_backbone_weights
from nowcasting.models.registry import build_model
from test.radar import (
    finalize_horizon_event_metrics,
    init_horizon_event_counts,
    parse_horizon_bins,
    update_horizon_event_counts,
)


def _model_args(device, input_length=2, total_length=4, height=32, width=32, model_name="PWVBirthGrowthNowcastNet"):
    ngf = 4
    return SimpleNamespace(
        model_name=model_name,
        input_length=input_length,
        total_length=total_length,
        img_height=height,
        img_width=width,
        img_ch=2,
        ngf=ngf,
        evo_ic=total_length - input_length,
        gen_oc=total_length - input_length,
        ic_feature=ngf * 10,
        intensity_scale=35.0,
        evo_base_channels=4,
        pwv_base_channels=4,
        fusion_channels=4,
        lead_time_embed_dim=4,
        pwv_source_type="cnn",
        frame_minutes=6.0,
        pwv_tendency_windows="6,12",
        pwv_tendency_mode="slope",
        birth_low_threshold=2.0,
        birth_high_threshold=10.0,
        growth_delta=5.0,
        birth_focal_alpha=0.75,
        birth_focal_gamma=2.0,
        lambda_birth=0.5,
        lambda_growth=0.5,
        lambda_positive_source=0.5,
        lambda_source_sparse=0.05,
        source_active_weight=4.0,
        source_inactive_weight=0.1,
        birth_loss_normalization="class_balanced",
        device=device,
    )


class BirthGrowthModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_loss_metrics_and_freeze(self):
        torch.manual_seed(2026)
        args = _model_args(self.device)
        model = build_model(args).to(self.device)
        model.freeze_radar_backbone()
        model.train()

        self.assertFalse(model.radar_evo_net.training)
        self.assertTrue(all(not p.requires_grad for p in model.radar_evo_net.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.pwv_source_net.parameters()))

        radar = torch.rand(1, args.total_length, 32, 32, 2, device=self.device) * 20.0
        pwv = torch.rand(1, args.total_length, 32, 32, device=self.device) * 80.0
        aux = model(radar, apply_pwv_control(pwv, "real"), return_aux=True)

        expected = (1, args.total_length - args.input_length, 1, 32, 32)
        self.assertEqual(tuple(aux["prediction"].shape), (1, 2, 32, 32, 1))
        self.assertEqual(tuple(aux["birth_probability"].shape), expected)
        self.assertEqual(tuple(aux["growth_probability"].shape), expected)
        self.assertEqual(tuple(aux["pwv_contribution"].shape), expected)
        self.assertTrue(torch.isfinite(aux["prediction"]).all().item())
        self.assertGreaterEqual(float(aux["pwv_contribution"].detach().min()), 0.0)
        self.assertGreaterEqual(float(aux["birth_probability"].detach().min()), 0.0)
        self.assertLessEqual(float(aux["birth_probability"].detach().max()), 1.0)

        target = radar[:, args.input_length :, :, :, 0]
        total_loss, parts = birth_growth_losses(aux, target, args)
        self.assertTrue(torch.isfinite(total_loss).item())
        total_loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.pwv_source_net.parameters()))
        self.assertTrue(all(p.grad is None for p in model.radar_evo_net.parameters()))
        self.assertEqual(set(parts), {
            "birth", "growth", "positive_source", "positive_source_active",
            "positive_source_inactive", "source_sparse", "birth_rate", "growth_rate"
        })

        accumulator = BirthGrowthAccumulator(bins=20)
        accumulator.update(aux, target, args)
        metrics = accumulator.finalize()
        self.assertIn("birth", metrics)
        self.assertIn("growth", metrics)
        self.assertGreater(metrics["birth"]["count"], 0)

    def test_pwv_controls(self):
        pwv = torch.arange(12).reshape(1, 3, 2, 2)
        self.assertTrue(torch.equal(apply_pwv_control(pwv, "real"), pwv))
        self.assertEqual(int(apply_pwv_control(pwv, "zero").sum()), 0)
        self.assertTrue(torch.equal(apply_pwv_control(pwv, "temporal_reverse")[:, 0], pwv[:, -1]))

    def test_frozen_protocol_dimensions_for_radar_and_pwv(self):
        """Exercise the exact 9-to-30 and 96x96 tensor contract with light channels."""
        torch.manual_seed(2026)
        radar = torch.rand(1, 39, 96, 96, 2, device=self.device)
        pwv = torch.rand(1, 39, 96, 96, device=self.device)
        with torch.no_grad():
            radar_args = _model_args(
                self.device, input_length=9, total_length=39, height=96, width=96,
                model_name="NowcastNet",
            )
            radar_model = build_model(radar_args).to(self.device).eval()
            radar_prediction = radar_model(radar)
            self.assertEqual(tuple(radar_prediction.shape), (1, 30, 96, 96, 1))
            self.assertTrue(torch.isfinite(radar_prediction).all().item())
            del radar_model, radar_prediction

            pwv_args = _model_args(self.device, input_length=9, total_length=39, height=96, width=96)
            pwv_model = build_model(pwv_args).to(self.device).eval()
            aux = pwv_model(radar, pwv, return_aux=True)
            self.assertEqual(tuple(aux["prediction"].shape), (1, 30, 96, 96, 1))
            self.assertEqual(tuple(aux["pwv_contribution"].shape), (1, 30, 1, 96, 96))
            self.assertTrue(torch.isfinite(aux["prediction"]).all().item())

            trigger_radar = radar[:, :29]
            trigger_pwv = pwv[:, :29]
            trigger_args = _model_args(
                self.device, input_length=9, total_length=29, height=96, width=96,
                model_name="PWVContrastiveTriggerNowcastNet",
            )
            trigger_args.pwv_intensity_scale = 80.0
            trigger_args.pwv_candidate_threshold = 0.5
            trigger_args.pwv_candidate_radius = 4
            trigger_model = build_model(trigger_args).to(self.device).eval()
            trigger_aux = trigger_model(trigger_radar, trigger_pwv, return_aux=True)
            self.assertEqual(tuple(trigger_aux["prediction"].shape), (1, 20, 96, 96, 1))
            self.assertEqual(tuple(trigger_aux["pwv_contribution"].shape), (1, 20, 1, 96, 96))
            self.assertTrue(torch.isfinite(trigger_aux["prediction"]).all().item())

    def test_radar_checkpoint_maps_into_birth_growth_backbone(self):
        radar_args = _model_args(self.device, model_name="NowcastNet")
        pwv_args = _model_args(self.device)
        # The radar baseline uses the historical fixed 32-channel evolution net;
        # checkpoint compatibility must be tested against that exact width.
        pwv_args.evo_base_channels = 32
        radar_model = build_model(radar_args)
        pwv_model = build_model(pwv_args)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "radar.ckpt"
            torch.save(radar_model.state_dict(), checkpoint)
            report = load_radar_backbone_weights(pwv_model, checkpoint, self.device)
        self.assertGreater(report["loaded_tensors"], 0)
        radar_value = radar_model.state_dict()["evo_net.inc.double_conv.0.weight"]
        pwv_value = pwv_model.state_dict()["radar_evo_net.inc.double_conv.0.weight"]
        self.assertTrue(torch.equal(radar_value, pwv_value))

    def test_contrastive_trigger_null_pwv_is_exact_radar_identity(self):
        torch.manual_seed(2026)
        radar_args = _model_args(
            self.device, input_length=2, total_length=4, height=32, width=32,
            model_name="NowcastNet",
        )
        trigger_args = _model_args(
            self.device, input_length=2, total_length=4, height=32, width=32,
            model_name="PWVContrastiveTriggerNowcastNet",
        )
        trigger_args.evo_base_channels = 32
        trigger_args.pwv_intensity_scale = 80.0
        trigger_args.pwv_candidate_threshold = 0.5
        trigger_args.pwv_candidate_radius = 2
        radar_model = build_model(radar_args).to(self.device).eval()
        trigger_model = build_model(trigger_args).to(self.device).eval()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "radar.ckpt"
            torch.save(radar_model.state_dict(), checkpoint)
            load_radar_backbone_weights(trigger_model, checkpoint, self.device)

        frames = torch.rand(1, 4, 32, 32, 2, device=self.device) * 20.0
        null_pwv = torch.zeros(1, 4, 32, 32, device=self.device)
        with torch.no_grad():
            torch.manual_seed(99)
            radar_prediction = radar_model(frames)
            torch.manual_seed(99)
            aux = trigger_model(frames, null_pwv, return_aux=True)

        self.assertTrue(torch.equal(aux["pwv_contribution"], torch.zeros_like(aux["pwv_contribution"])))
        self.assertTrue(torch.equal(aux["pwv_birth_evidence"], torch.zeros_like(aux["pwv_birth_evidence"])))
        self.assertTrue(torch.equal(aux["pwv_growth_evidence"], torch.zeros_like(aux["pwv_growth_evidence"])))
        self.assertTrue(torch.allclose(aux["prediction"], radar_prediction, atol=1e-6, rtol=1e-6))

    def test_horizon_event_metrics_use_disjoint_lead_bins(self):
        bins = parse_horizon_bins("0-1,1-2,2-3")
        counts = init_horizon_event_counts([10.0], bins)
        prediction = torch.zeros(1, 30, 1, 1)
        target = torch.zeros_like(prediction)
        prediction[:, :10] = 10.0
        target[:, :20] = 10.0

        update_horizon_event_counts(counts, prediction, target, [10.0], 6.0, bins)
        metrics = finalize_horizon_event_metrics(counts)

        self.assertEqual(metrics["0h-1h"]["10.0"]["csi"], 1.0)
        self.assertEqual(metrics["1h-2h"]["10.0"]["csi"], 0.0)
        self.assertTrue(np.isnan(metrics["2h-3h"]["10.0"]["csi"]))


class StrictDatasetSmokeTest(unittest.TestCase):
    def test_manifest_continuity_pairing_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            radar_day = root / "radar" / "2025-07-01"
            pwv_day = root / "pwv" / "2025-07-01"
            radar_day.mkdir(parents=True)
            pwv_day.mkdir(parents=True)
            names = [
                "2025-07-01-00-00-00.png",
                "2025-07-01-00-06-00.png",
                "2025-07-01-00-12-00.png",
                "2025-07-01-00-18-00.png",
            ]
            for index, name in enumerate(names):
                image = Image.fromarray(np.full((32, 32), index * 10, dtype=np.uint8))
                image.save(radar_day / name)
                image.save(pwv_day / name)

            manifest = root / "split.json"
            manifest.write_text(
                json.dumps({"splits": {"train": ["2025-07-01"], "val": [], "test": []}}),
                encoding="utf-8",
            )
            dataset = PngSequenceDataset(
                data_root=root / "radar",
                pwv_root=root / "pwv",
                split="train",
                split_manifest=manifest,
                input_length=2,
                total_length=4,
                img_height=32,
                img_width=32,
                require_contiguous=True,
                strict_pwv=True,
            )
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(tuple(sample["radar_frames"].shape), (4, 32, 32, 2))
            self.assertEqual(tuple(sample["pwv_frames"].shape), (4, 32, 32))
            provenance = dataset.provenance()
            self.assertEqual(provenance["samples"], 1)
            self.assertEqual(len(provenance["sample_sha256"]), 64)

            (pwv_day / names[-1]).unlink()
            with self.assertRaises(FileNotFoundError):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
