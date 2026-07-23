"""Tests for paired day-cluster comparison of signed calibrator controls."""

import unittest

from report.protocol_compare_signed import paired_day_bootstrap


def record(sample_id, case_name, hit, miss, false_alarm):
    return {
        "sample_id": sample_id,
        "case_name": case_name,
        "model_events": {
            "10.0": {
                "hit": hit,
                "miss": miss,
                "false_alarm": false_alarm,
            }
        },
    }


class SignedReportTest(unittest.TestCase):
    def test_paired_day_bootstrap_detects_positive_delta(self):
        better = [
            record("a", "day1", 8, 1, 1),
            record("b", "day1", 7, 2, 1),
            record("c", "day2", 9, 1, 1),
            record("d", "day2", 8, 1, 2),
        ]
        worse = [
            record("a", "day1", 5, 4, 2),
            record("b", "day1", 4, 5, 2),
            record("c", "day2", 6, 4, 2),
            record("d", "day2", 5, 4, 3),
        ]
        result = paired_day_bootstrap(
            better, worse, "10.0", repetitions=200, seed=2026
        )
        self.assertEqual(result["n_cases"], 2)
        self.assertGreater(result["mean"], 0.0)
        self.assertGreater(result["ci95"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
