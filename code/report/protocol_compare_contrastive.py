import argparse
import json
import math
import statistics
from pathlib import Path


CORE_VARIANTS = ("radar", "pwv_real", "pwv_null", "pwv_temporal_reverse")
DIAGNOSTIC_VARIANTS = ("pwv_level_only", "pwv_spatial_shift")


def nested(obj, *keys):
    for key in keys:
        if obj is None:
            return None
        obj = obj.get(key)
    return obj


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_manifest_hash(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(iter(payload.values()))["sample_sha256"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for seed_dir in sorted(Path(args.run_root).glob("seed_*")):
        core_paths = {
            name: seed_dir / "results" / name / "metrics.json" for name in CORE_VARIANTS
        }
        if not all(path.exists() for path in core_paths.values()):
            continue
        metric_paths = dict(core_paths)
        for name in DIAGNOSTIC_VARIANTS:
            path = seed_dir / "results" / name / "metrics.json"
            if path.exists():
                metric_paths[name] = path
        metrics = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in metric_paths.items()
        }
        reverse_scope = metrics["pwv_temporal_reverse"].get("pwv_control_scope")
        if reverse_scope != "observed_input_only":
            raise ValueError(
                "{} has a stale temporal-reverse result; rerun the observed-input-only control".format(
                    seed_dir.name
                )
            )
        hashes = {
            name: load_manifest_hash(seed_dir / "results" / name / "data_manifest.json")
            for name in metric_paths
        }
        if len(set(hashes.values())) != 1:
            raise ValueError("Sample identity mismatch in {}".format(seed_dir.name))
        samples = {metrics[name].get("samples") for name in metric_paths}
        if len(samples) != 1:
            raise ValueError("Sample count mismatch in {}".format(seed_dir.name))

        row = {
            "seed": int(seed_dir.name.split("_")[-1]),
            "samples": next(iter(samples)),
            "sample_sha256": next(iter(hashes.values())),
        }
        for name in metric_paths:
            row["{}_mae".format(name)] = nested(metrics[name], "model", "mae")
            row["{}_rmse".format(name)] = nested(metrics[name], "model", "rmse")
            for threshold in ("10.0", "20.0"):
                key = threshold.replace(".", "p")
                for event_metric in ("csi", "pod", "far", "bias"):
                    row["{}_{}_{}".format(name, event_metric, key)] = nested(
                        metrics[name], "event_metrics", "model", threshold, event_metric
                    )
            if name != "radar":
                row["{}_birth_pr_auc".format(name)] = nested(
                    metrics[name], "birth_growth", "birth", "pr_auc_histogram"
                )
                row["{}_growth_pr_auc".format(name)] = nested(
                    metrics[name], "birth_growth", "growth", "pr_auc_histogram"
                )

        for metric in ("mae", "rmse", "csi_10p0", "csi_20p0"):
            real_key = "pwv_real_{}".format(metric)
            for reference in (
                "radar",
                "pwv_null",
                "pwv_temporal_reverse",
                "pwv_level_only",
                "pwv_spatial_shift",
            ):
                reference_key = "{}_{}".format(reference, metric)
                if finite(row.get(real_key)) and finite(row.get(reference_key)):
                    row["delta_{}_real_minus_{}".format(metric, reference)] = (
                        row[real_key] - row[reference_key]
                    )
        rows.append(row)

    if not rows:
        raise ValueError("No complete contrastive-control runs under {}".format(args.run_root))

    aggregate = {}
    numeric_keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if key not in ("seed", "samples") and finite(value)
    })
    for key in numeric_keys:
        values = [row[key] for row in rows if finite(row.get(key))]
        aggregate[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    output = {
        "protocol": "pwv_contrastive_trigger_pilot",
        "evaluation_role": "development pilot; not an untouched final test",
        "pwv_control_scope": "observed_input_only",
        "paired_seed_results": rows,
        "aggregate_across_seeds": aggregate,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
