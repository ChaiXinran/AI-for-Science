import argparse
import json
import math
import statistics
from pathlib import Path


def nested(obj, *keys):
    for key in keys:
        if obj is None:
            return None
        obj = obj.get(key)
    return obj


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for seed_dir in sorted(Path(args.run_root).glob("seed_*")):
        radar_path = seed_dir / "results" / "radar" / "metrics.json"
        zero_path = seed_dir / "results" / "birth_growth_zero" / "metrics.json"
        bg_path = seed_dir / "results" / "birth_growth" / "metrics.json"
        if not radar_path.exists() or not zero_path.exists() or not bg_path.exists():
            continue
        radar = json.loads(radar_path.read_text(encoding="utf-8"))
        zero = json.loads(zero_path.read_text(encoding="utf-8"))
        bg = json.loads(bg_path.read_text(encoding="utf-8"))
        radar_data = json.loads((seed_dir / "results" / "radar" / "data_manifest.json").read_text(encoding="utf-8"))
        zero_data = json.loads((seed_dir / "results" / "birth_growth_zero" / "data_manifest.json").read_text(encoding="utf-8"))
        bg_data = json.loads((seed_dir / "results" / "birth_growth" / "data_manifest.json").read_text(encoding="utf-8"))
        radar_hash = next(iter(radar_data.values()))["sample_sha256"]
        zero_hash = next(iter(zero_data.values()))["sample_sha256"]
        bg_hash = next(iter(bg_data.values()))["sample_sha256"]
        if len({radar_hash, zero_hash, bg_hash}) != 1:
            raise ValueError("Sample identity mismatch in {}".format(seed_dir.name))
        if radar.get("samples") != bg.get("samples"):
            raise ValueError("Sample count mismatch in {}".format(seed_dir.name))
        row = {
            "seed": int(seed_dir.name.split("_")[-1]),
            "samples": radar.get("samples"),
            "sample_sha256": radar_hash,
            "radar_mae": nested(radar, "model", "mae"),
            "birth_growth_mae": nested(bg, "model", "mae"),
            "radar_rmse": nested(radar, "model", "rmse"),
            "zero_pwv_mae": nested(zero, "model", "mae"),
            "zero_pwv_rmse": nested(zero, "model", "rmse"),
            "birth_growth_rmse": nested(bg, "model", "rmse"),
            "zero_pwv_birth_pr_auc": nested(zero, "birth_growth", "birth", "pr_auc_histogram"),
            "birth_pr_auc": nested(bg, "birth_growth", "birth", "pr_auc_histogram"),
            "zero_pwv_growth_pr_auc": nested(zero, "birth_growth", "growth", "pr_auc_histogram"),
            "growth_pr_auc": nested(bg, "birth_growth", "growth", "pr_auc_histogram"),
            "birth_pod": nested(bg, "birth_growth", "birth", "recall_pod"),
            "birth_far": nested(bg, "birth_growth", "birth", "false_alarm_ratio"),
            "birth_positives": nested(bg, "birth_growth", "birth", "positives"),
            "birth_positive_rate": nested(bg, "birth_growth", "birth", "positive_rate"),
            "growth_pod": nested(bg, "birth_growth", "growth", "recall_pod"),
            "growth_far": nested(bg, "birth_growth", "growth", "false_alarm_ratio"),
            "growth_positives": nested(bg, "birth_growth", "growth", "positives"),
            "growth_positive_rate": nested(bg, "birth_growth", "growth", "positive_rate"),
            "positive_source_mae_active": nested(bg, "birth_growth", "positive_source_mae_active"),
        }
        if row["birth_pr_auc"] is not None and row["zero_pwv_birth_pr_auc"] is not None:
            row["birth_pr_auc_delta_vs_zero_pwv"] = row["birth_pr_auc"] - row["zero_pwv_birth_pr_auc"]
        if row["growth_pr_auc"] is not None and row["zero_pwv_growth_pr_auc"] is not None:
            row["growth_pr_auc_delta_vs_zero_pwv"] = row["growth_pr_auc"] - row["zero_pwv_growth_pr_auc"]
        for threshold in ("10.0", "20.0"):
            key = threshold.replace(".", "p")
            row["radar_csi_{}".format(key)] = nested(radar, "event_metrics", "model", threshold, "csi")
            row["zero_pwv_csi_{}".format(key)] = nested(zero, "event_metrics", "model", threshold, "csi")
            row["birth_growth_csi_{}".format(key)] = nested(bg, "event_metrics", "model", threshold, "csi")
        rows.append(row)
    if not rows:
        raise ValueError("No complete seed pairs found under {}".format(args.run_root))
    numeric_keys = sorted(
        key for key, value in rows[0].items()
        if key not in ("seed", "samples") and finite_number(value)
    )
    aggregate = {}
    for key in numeric_keys:
        values = [row[key] for row in rows if finite_number(row.get(key))]
        if values:
            aggregate[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    for metric in ("mae", "rmse", "csi_10p0", "csi_20p0"):
        radar_key = "radar_{}".format(metric)
        bg_key = "birth_growth_{}".format(metric)
        values = [row[bg_key] - row[radar_key] for row in rows if finite_number(row.get(bg_key)) and finite_number(row.get(radar_key))]
        if values:
            aggregate["paired_delta_{}_minus_radar".format(metric)] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        zero_key = "zero_pwv_{}".format(metric)
        zero_values = [row[bg_key] - row[zero_key] for row in rows if finite_number(row.get(bg_key)) and finite_number(row.get(zero_key))]
        if zero_values:
            aggregate["paired_delta_{}_minus_zero_pwv".format(metric)] = {
                "mean": statistics.mean(zero_values),
                "std": statistics.stdev(zero_values) if len(zero_values) > 1 else 0.0,
                "n": len(zero_values),
            }
    output = {
        "protocol": "pwv_birth_growth_v1",
        "paired_seed_results": rows,
        "aggregate_across_seeds": aggregate,
        "interpretation_note": "Seed-level summaries are not event-bootstrap confidence intervals.",
    }
    output = sanitize(output)
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
