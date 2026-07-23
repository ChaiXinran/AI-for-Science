"""Compare radar-only, aligned-PWV and displaced-PWV latent models."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report.protocol_compare_signed import (
    paired_day_bootstrap,
    threshold_key,
)


VARIANTS = ("radar_only", "pwv_aligned", "pwv_displaced")


def load_variant(root, name):
    result = root / "results" / name
    metrics_path = result / "metrics.json"
    records_path = result / "eventwise_records.json"
    manifest_path = result / "data_manifest.json"
    if not metrics_path.exists() or not records_path.exists():
        raise FileNotFoundError(
            "Missing metrics/eventwise records for {}".format(name)
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    sample_sha256 = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest:
            split_payload = next(iter(manifest.values()))
            sample_sha256 = split_payload.get("sample_sha256")
    return {
        "metrics": metrics,
        "records": records,
        "sample_sha256": sample_sha256,
    }


def summarize(payload, thresholds):
    metrics = payload["metrics"]
    result = {
        "samples": metrics["samples"],
        "sample_sha256": payload["sample_sha256"],
        "mae": metrics["model"]["mae"],
        "rmse": metrics["model"]["rmse"],
        "bias": metrics["model"]["bias"],
        "event_metrics": {},
        "horizon_event_metrics": metrics.get("horizon_event_metrics", {}).get(
            "model", {}
        ),
    }
    for threshold in thresholds:
        key = threshold_key(metrics, threshold)
        result["event_metrics"]["{:g}".format(threshold)] = metrics[
            "event_metrics"
        ]["model"][key]
    return result


def optional_delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="10,20")
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--minimum_csi_delta", type=float, default=0.003)
    parser.add_argument("--maximum_far_delta", type=float, default=0.005)
    parser.add_argument(
        "--maximum_relative_mae_increase", type=float, default=0.005
    )
    args = parser.parse_args()
    root = Path(args.seed_root)
    thresholds = [
        float(item) for item in args.thresholds.split(",") if item.strip()
    ]
    loaded = {name: load_variant(root, name) for name in VARIANTS}
    hashes = {
        payload["sample_sha256"]
        for payload in loaded.values()
        if payload["sample_sha256"] is not None
    }
    if len(hashes) > 1:
        raise ValueError("Evaluation sample hashes differ: {}".format(hashes))

    summary = {
        "protocol": "pwv_latent_state_fusion_pilot",
        "seed_root": str(root),
        "thresholds": thresholds,
        "variants": {
            name: summarize(payload, thresholds)
            for name, payload in loaded.items()
        },
        "paired_comparisons": {},
        "gate": {
            "minimum_csi_delta": args.minimum_csi_delta,
            "maximum_far_delta": args.maximum_far_delta,
            "maximum_relative_mae_increase": args.maximum_relative_mae_increase,
        },
    }
    for reference in ("radar_only", "pwv_displaced"):
        comparison = {"by_threshold": {}}
        for threshold in thresholds:
            label = "{:g}".format(threshold)
            aligned_event = summary["variants"]["pwv_aligned"][
                "event_metrics"
            ][label]
            reference_event = summary["variants"][reference][
                "event_metrics"
            ][label]
            event_key = threshold_key(
                loaded["pwv_aligned"]["metrics"], threshold
            )
            comparison["by_threshold"][label] = {
                "csi_delta": optional_delta(
                    aligned_event["csi"], reference_event["csi"]
                ),
                "far_delta": optional_delta(
                    aligned_event["far"], reference_event["far"]
                ),
                "day_cluster_bootstrap_csi_delta": paired_day_bootstrap(
                    loaded["pwv_aligned"]["records"],
                    loaded[reference]["records"],
                    event_key,
                    args.bootstrap_repetitions,
                    args.bootstrap_seed,
                ),
            }
        summary["paired_comparisons"][
            "pwv_aligned_minus_{}".format(reference)
        ] = comparison

    radar_mae = summary["variants"]["radar_only"]["mae"]
    aligned_mae = summary["variants"]["pwv_aligned"]["mae"]
    relative_mae = (aligned_mae - radar_mae) / max(abs(radar_mae), 1e-12)
    checks = [relative_mae <= args.maximum_relative_mae_increase]
    for reference in ("radar_only", "pwv_displaced"):
        for threshold in thresholds:
            row = summary["paired_comparisons"][
                "pwv_aligned_minus_{}".format(reference)
            ]["by_threshold"]["{:g}".format(threshold)]
            checks.append(
                row["csi_delta"] is not None
                and row["csi_delta"] >= args.minimum_csi_delta
            )
            if reference == "radar_only" and row["far_delta"] is not None:
                checks.append(row["far_delta"] <= args.maximum_far_delta)
    summary["gate"].update(
        {
            "relative_mae_increase_vs_radar": relative_mae,
            "point_estimate_pass": bool(all(checks)),
            "interpretation": (
                "One seed is development evidence only. A point-estimate pass "
                "promotes to three-seed replication; replication additionally "
                "requires positive day-cluster bootstrap lower bounds."
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
