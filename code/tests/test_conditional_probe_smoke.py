import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.pwv_conditional_probe import (
    ConditionalEventProbe,
    _event_targets,
    _pwv_features,
    cross_event_indices,
    paired_day_bootstrap,
)


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


if __name__ == "__main__":
    unittest.main()
