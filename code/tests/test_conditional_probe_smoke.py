import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.pwv_conditional_probe import (
    ConditionalEventProbe,
    _event_targets,
    _pwv_features,
    cross_event_indices,
    paired_day_bootstrap,
)
from diagnostics.pwv_preconditioning_probe import (
    PRIMARY_STRATUM,
    build_strata,
    causal_pwv_features,
)
from nowcasting.data_provider.custom_png import PngSequenceDataset


class ConditionalProbeSmokeTest(unittest.TestCase):
    def test_shapes_parameter_matching_and_pwv_sensitivity(self):
        torch.manual_seed(3)
        kwargs = dict(
            radar_channels=16,
            pwv_channels=6,
            hidden_channels=8,
            lead_count=20,
            threshold_count=2,
        )
        radar_probe = ConditionalEventProbe(**kwargs)
        pwv_probe = ConditionalEventProbe(**kwargs)
        self.assertEqual(
            sum(p.numel() for p in radar_probe.parameters()),
            sum(p.numel() for p in pwv_probe.parameters()),
        )
        radar = torch.randn(2, 16, 12, 12)
        pwv = torch.randn(2, 6, 12, 12)
        radar_logits = radar_probe(radar, pwv, use_pwv=False)
        real_logits = pwv_probe(radar, pwv, use_pwv=True)
        shifted_logits = pwv_probe(
            radar,
            torch.roll(pwv, shifts=(6, 6), dims=(-2, -1)),
            use_pwv=True,
        )
        self.assertEqual(tuple(real_logits.shape), (2, 20, 2, 12, 12))
        self.assertTrue(torch.equal(
            radar_logits,
            radar_probe(radar, torch.zeros_like(pwv), use_pwv=False),
        ))
        self.assertGreater(
            float((real_logits - shifted_logits).abs().mean().detach()), 0
        )

    def test_observed_pwv_features_and_future_targets(self):
        args = SimpleNamespace(
            input_length=9,
            pwv_intensity_scale=80.0,
        )
        pwv = torch.rand(2, 29, 96, 96) * 80
        original = _pwv_features(pwv, args, (12, 12))
        pwv[:, 9:] = 999
        changed_future = _pwv_features(pwv, args, (12, 12))
        self.assertTrue(torch.equal(original, changed_future))
        target = torch.zeros(2, 20, 96, 96)
        target[:, 0, :8, :8] = 12
        target[:, 10, 8:16, 8:16] = 22
        events = _event_targets(target, [10.0, 20.0], (12, 12))
        self.assertEqual(tuple(events.shape), (2, 20, 2, 12, 12))
        self.assertEqual(int(events[:, 0, 0, 0, 0].sum()), 2)
        self.assertEqual(int(events[:, 10, 1, 1, 1].sum()), 2)

    def test_day_cluster_bootstrap(self):
        left, right = [], []
        for case in ("a", "b", "c"):
            for index in range(2):
                sample_id = "{}-{}".format(case, index)
                left.append({
                    "sample_id": sample_id,
                    "case_name": case,
                    "events": {"task": {"hit": 8, "miss": 2, "false_alarm": 1}},
                })
                right.append({
                    "sample_id": sample_id,
                    "case_name": case,
                    "events": {"task": {"hit": 5, "miss": 5, "false_alarm": 3}},
                })
        result = paired_day_bootstrap(left, right, "task", 100, 4)
        self.assertGreater(result["ci95"][0], 0)

    def test_cross_event_mapping_never_uses_same_case(self):
        cases = ["a", "a", "b", "c", "c", "c"]
        mapping = cross_event_indices(cases)
        for index, donor in enumerate(mapping.tolist()):
            self.assertNotEqual(cases[index], cases[donor])

    def test_causal_native_pwv_history_never_uses_future_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            radar_root = root / "rain"
            pwv_root = root / "pwv"
            radar_day = radar_root / "202505" / "20250502"
            radar_day.mkdir(parents=True)
            start = datetime(2025, 5, 2, 0, 0)
            for step in range(29):
                stamp = start + timedelta(minutes=6 * step)
                Image.fromarray(
                    np.full((8, 8), 255 - step, dtype=np.uint8)
                ).save(radar_day / (stamp.strftime("%Y-%m-%d-%H-%M-%S") + ".png"))

            pwv_start = datetime(2025, 5, 1, 21, 30)
            pwv_end = start + timedelta(minutes=6 * 28)
            stamp = pwv_start
            while stamp <= pwv_end:
                day = pwv_root / stamp.strftime("%Y%m") / stamp.strftime("%Y%m%d")
                day.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    np.full((8, 8), stamp.minute, dtype=np.uint8)
                ).save(day / (stamp.strftime("%Y-%m-%d-%H-%M-%S") + ".png"))
                stamp += timedelta(minutes=6)

            dataset = PngSequenceDataset(
                data_root=radar_root,
                pwv_root=pwv_root,
                split="all",
                input_length=9,
                total_length=29,
                img_height=32,
                img_width=32,
                require_contiguous=True,
                strict_pwv=True,
                pwv_history_minutes=180,
                pwv_anchor_minutes=30,
                pwv_invert=True,
                pwv_intensity_scale=80,
            )
            sample = dataset[0]
            self.assertEqual(
                tuple(sample["pwv_history_frames"].shape), (7, 32, 32)
            )
            self.assertEqual(
                sample["pwv_history_start_file"],
                "2025-05-01-21-30-00.png",
            )
            self.assertEqual(
                sample["pwv_history_end_file"],
                "2025-05-02-00-30-00.png",
            )
            issue = datetime.strptime(
                dataset.windows[0][8].stem, "%Y-%m-%d-%H-%M-%S"
            )
            history_end = datetime.strptime(
                Path(sample["pwv_history_end_file"]).stem,
                "%Y-%m-%d-%H-%M-%S",
            )
            self.assertLessEqual(history_end, issue)

    def test_long_features_and_observed_radar_strata(self):
        history = torch.arange(7.0).view(1, 7, 1, 1).expand(2, 7, 4, 4)
        climatology = torch.zeros(1, 1, 4, 4)
        features = causal_pwv_features(history, climatology, 80.0)
        self.assertEqual(tuple(features.shape), (2, 6, 4, 4))
        self.assertTrue(torch.all(features[:, 3] > 0))

        cache = {
            "target": torch.zeros(2, 20, 2, 4, 4, dtype=torch.uint8),
            "observed_radar_tiles": torch.zeros(2, 9, 4, 4),
        }
        cache["observed_radar_tiles"][:, :3] = 1
        cache["observed_radar_tiles"][:, -1] = 2
        masks = build_strata(cache, [10.0, 20.0])
        self.assertTrue(masks[PRIMARY_STRATUM].all())
        self.assertFalse(masks["radar_quiet"].any())


if __name__ == "__main__":
    unittest.main()
