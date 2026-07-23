"""Compare signed-calibrator controls with paired day-cluster bootstrap."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = (
    "static_real",
    "static_null",
    "static_level",
    "static_reverse",
    "static_shift",
    "spatial_control",
    "tendency_real",
)


def load_variant(root, name):
    result = root / "results" / name
    metrics_path = result / "metrics.json"
    records_path = result / "eventwise_records.json"
    if not metrics_path.exists() or not records_path.exists():
        return None
    return {
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "records": json.loads(records_path.read_text(encoding="utf-8")),
    }


def threshold_key(metrics, threshold):
    available = metrics["event_metrics"]["model"]
    for key in available:
        if abs(float(key) - float(threshold)) < 1e-6:
            return key
    raise KeyError("Threshold {} missing from {}".format(threshold, sorted(available)))


def paired_records(left, right):
    left_map = {row["sample_id"]: row for row in left}
    right_map = {row["sample_id"]: row for row in right}
    if set(left_map) != set(right_map):
        missing_left = sorted(set(right_map) - set(left_map))
        missing_right = sorted(set(left_map) - set(right_map))
        raise ValueError(
            "Sample mismatch: missing_left={} missing_right={}".format(
                missing_left[:3], missing_right[:3]
            )
        )
    return [(left_map[key], right_map[key]) for key in sorted(left_map)]


def aggregate_csi(records, event_key):
    hit = miss = false_alarm = 0
    for row in records:
        event = row["model_events"][event_key]
        hit += int(event["hit"])
        miss += int(event["miss"])
        false_alarm += int(event["false_alarm"])
    denominator = hit + miss + false_alarm
    return hit / denominator if denominator else None


def paired_day_bootstrap(left, right, event_key, repetitions, seed):
    pairs = paired_records(left, right)
    by_case = defaultdict(list)
    for left_row, right_row in pairs:
        if left_row["case_name"] != right_row["case_name"]:
            raise ValueError("Paired records disagree on case_name.")
        by_case[left_row["case_name"]].append((left_row, right_row))
    cases = sorted(by_case)
    if not cases:
        return {"n_cases": 0, "repetitions": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        selected = rng.choice(cases, size=len(cases), replace=True)
        sampled_left = []
        sampled_right = []
        for case in selected:
            for left_row, right_row in by_case[case]:
                sampled_left.append(left_row)
                sampled_right.append(right_row)
        left_csi = aggregate_csi(sampled_left, event_key)
        right_csi = aggregate_csi(sampled_right, event_key)
        if left_csi is not None and right_csi is not None:
            deltas.append(left_csi - right_csi)
    if not deltas:
        return {
            "n_cases": len(cases),
            "repetitions": 0,
            "mean": None,
            "ci95": [None, None],
        }
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "n_cases": len(cases),
        "repetitions": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def summarize_variant(payload, thresholds):
    metrics = payload["metrics"]
    summary = {
        "samples": metrics["samples"],
        "sample_sha256": None,
        "mae": metrics["model"]["mae"],
        "rmse": metrics["model"]["rmse"],
        "bias": metrics["model"]["bias"],
        "event_metrics": {},
    }
    for threshold in thresholds:
        key = threshold_key(metrics, threshold)
        summary["event_metrics"]["{:g}".format(threshold)] = metrics[
            "event_metrics"
        ]["model"][key]
    manifest_path = None
    # The dataset provenance is written beside metrics by the evaluator.
    # Keep this optional for backward-compatible smoke outputs.
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="10,20")
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--minimum_csi_delta", type=float, default=0.003)
    parser.add_argument("--maximum_far_delta", type=float, default=0.005)
    parser.add_argument("--maximum_relative_mae_increase", type=float, default=0.005)
    args = parser.parse_args()

    root = Path(args.seed_root)
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    loaded = {name: load_variant(root, name) for name in VARIANTS}
    missing_required = [
        name for name in ("static_real", "static_null", "spatial_control")
        if loaded[name] is None
    ]
    if missing_required:
        raise FileNotFoundError("Missing required result variants: {}".format(missing_required))

    summary = {
        "protocol": "pwv_signed_calibrator_pilot",
        "seed_root": str(root),
        "thresholds": thresholds,
        "variants": {
            name: summarize_variant(payload, thresholds)
            for name, payload in loaded.items()
            if payload is not None
        },
        "paired_comparisons": {},
        "gate": {
            "minimum_csi_delta": args.minimum_csi_delta,
            "maximum_far_delta": args.maximum_far_delta,
            "maximum_relative_mae_increase": args.maximum_relative_mae_increase,
        },
    }
    for reference in ("static_null", "spatial_control"):
        comparison = {"by_threshold": {}}
        for threshold in thresholds:
            label = "{:g}".format(threshold)
            real_event = summary["variants"]["static_real"]["event_metrics"][label]
            ref_event = summary["variants"][reference]["event_metrics"][label]
            key = threshold_key(loaded["static_real"]["metrics"], threshold)
            comparison["by_threshold"][label] = {
                "csi_delta": real_event["csi"] - ref_event["csi"],
                "far_delta": (
                    real_event["far"] - ref_event["far"]
                    if real_event["far"] is not None and ref_event["far"] is not None
                    else None
                ),
                "day_cluster_bootstrap_csi_delta": paired_day_bootstrap(
                    loaded["static_real"]["records"],
                    loaded[reference]["records"],
                    key,
                    args.bootstrap_repetitions,
                    args.bootstrap_seed,
                ),
            }
        summary["paired_comparisons"]["static_real_minus_{}".format(reference)] = comparison

    null_mae = summary["variants"]["static_null"]["mae"]
    real_mae = summary["variants"]["static_real"]["mae"]
    relative_mae = (real_mae - null_mae) / max(abs(null_mae), 1e-12)
    point_checks = []
    for reference in ("static_null", "spatial_control"):
        for threshold in thresholds:
            row = summary["paired_comparisons"][
                "static_real_minus_{}".format(reference)
            ]["by_threshold"]["{:g}".format(threshold)]
            point_checks.append(row["csi_delta"] >= args.minimum_csi_delta)
            if reference == "static_null" and row["far_delta"] is not None:
                point_checks.append(row["far_delta"] <= args.maximum_far_delta)
    point_checks.append(relative_mae <= args.maximum_relative_mae_increase)
    summary["gate"].update(
        {
            "relative_mae_increase_vs_null": relative_mae,
            "point_estimate_pass": bool(all(point_checks)),
            "interpretation": (
                "A one-seed point-estimate pass only promotes to replication; "
                "it is not paper evidence. Replication additionally requires "
                "positive day-cluster bootstrap lower bounds."
            ),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
