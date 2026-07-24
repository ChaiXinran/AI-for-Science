"""Create the four locked radar-only failure-attribution figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REGIMES = ("translation", "rapid_growth", "rapid_decay", "birth", "split_merge")
REGIME_LABELS = {
    "translation": "Translation",
    "rapid_growth": "Rapid growth",
    "rapid_decay": "Rapid decay",
    "birth": "Birth",
    "split_merge": "Split/merge",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Plot radar object-failure attribution.")
    parser.add_argument("--input", required=True, help="failure_attribution.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--change_fraction", type=float, default=0.4)
    parser.add_argument("--primary_horizon", default="1h-2h")
    parser.add_argument("--minimum_nonadvective_miss_share", type=float, default=0.2)
    return parser


def change_key(value):
    return "{:g}".format(float(value))


def threshold_sort_key(value):
    return float(value)


def load_values(payload, change_fraction):
    summary = payload["summary"]["thresholds"]
    key = change_key(change_fraction)
    values = {}
    for threshold, by_change in summary.items():
        if key not in by_change:
            raise ValueError("Change fraction {} not found for threshold {}.".format(key, threshold))
        values[threshold] = by_change[key]
    return values


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_object_counts(values, output_dir):
    thresholds = sorted(values, key=threshold_sort_key)
    horizons = list(next(iter(values.values())).keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4), squeeze=False)
    x = np.arange(len(thresholds))
    width = 0.16
    for axis, horizon in zip(axes[0], horizons):
        for index, regime in enumerate(REGIMES):
            counts = [values[threshold][horizon]["regimes"][regime]["observed"] for threshold in thresholds]
            axis.bar(x + (index - 2) * width, counts, width, label=REGIME_LABELS[regime])
        axis.set_title(horizon)
        axis.set_xticks(x, ["{} mm/h".format(item) for item in thresholds])
        axis.set_ylabel("Observed objects")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, output_dir / "01_object_counts_by_horizon.png")


def plot_regime_skill(values, output_dir):
    thresholds = sorted(values, key=threshold_sort_key)
    horizons = list(next(iter(values.values())).keys())
    fig, axes = plt.subplots(len(thresholds), len(horizons), figsize=(6 * len(horizons), 3.7 * len(thresholds)), squeeze=False)
    metrics = ("object_pod", "object_far", "strict_type_csi")
    labels = ("POD", "FAR", "Strict CSI")
    x = np.arange(len(REGIMES))
    width = 0.24
    for row, threshold in enumerate(thresholds):
        for col, horizon in enumerate(horizons):
            axis = axes[row, col]
            regimes = values[threshold][horizon]["regimes"]
            for index, (metric, label) in enumerate(zip(metrics, labels)):
                vals = [
                    np.nan if regimes[regime][metric] is None else regimes[regime][metric]
                    for regime in REGIMES
                ]
                axis.bar(x + (index - 1) * width, vals, width, label=label)
            axis.set_title("{} mm/h, {}".format(threshold, horizon))
            axis.set_xticks(x, [REGIME_LABELS[item] for item in REGIMES], rotation=25, ha="right")
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, output_dir / "02_regime_pod_far_csi.png")


def plot_oracle_gains(values, output_dir):
    thresholds = sorted(values, key=threshold_sort_key)
    horizons = list(next(iter(values.values())).keys())
    oracle_names = ("displacement", "intensity", "birth_existence", "existence")
    oracle_labels = ("Displacement", "Intensity", "Birth existence", "Full existence")
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4), squeeze=False)
    x = np.arange(len(thresholds))
    width = 0.19
    for axis, horizon in zip(axes[0], horizons):
        for index, (oracle, label) in enumerate(zip(oracle_names, oracle_labels)):
            vals = [
                values[threshold][horizon]["oracles"][oracle]["csi_delta_vs_original"]
                for threshold in thresholds
            ]
            vals = [np.nan if value is None else value for value in vals]
            axis.bar(x + (index - 1.5) * width, vals, width, label=label)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(horizon)
        axis.set_xticks(x, ["{} mm/h".format(item) for item in thresholds])
        axis.set_ylabel("CSI oracle gain")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    _save(fig, output_dir / "03_oracle_csi_gains.png")


def build_miss_source_summary(values, primary_horizon, minimum_share):
    rows = []
    for threshold in sorted(values, key=threshold_sort_key):
        if primary_horizon not in values[threshold]:
            continue
        regimes = values[threshold][primary_horizon]["regimes"]
        missed = {regime: int(regimes[regime]["missed"]) for regime in REGIMES}
        total = sum(missed.values())
        nonadvective = missed["birth"] + missed["rapid_growth"]
        rows.append(
            {
                "threshold": float(threshold),
                "horizon": primary_horizon,
                "missed_objects": total,
                "missed_by_regime": missed,
                "birth_growth_missed": nonadvective,
                "birth_growth_miss_share": float(nonadvective) / total if total else None,
                "passes_share_gate": bool(total and float(nonadvective) / total >= minimum_share),
            }
        )
    valid = [row for row in rows if row["birth_growth_miss_share"] is not None]
    return {
        "minimum_nonadvective_miss_share": minimum_share,
        "rows": rows,
        "thresholds_with_data": len(valid),
        "thresholds_passing": sum(row["passes_share_gate"] for row in valid),
        "provisional_gate": (
            "pass"
            if valid and sum(row["passes_share_gate"] for row in valid) >= max(1, len(valid) // 2 + len(valid) % 2)
            else "fail"
        ),
        "warning": "This is provisional until validation and test event-cluster intervals agree.",
    }


def plot_miss_sources(summary, output_dir):
    rows = summary["rows"]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(rows), dtype="float64")
    for regime in REGIMES:
        values = np.array([row["missed_by_regime"][regime] for row in rows], dtype="float64")
        axis.bar(x, values, bottom=bottom, label=REGIME_LABELS[regime])
        bottom += values
    axis.set_xticks(x, ["{:g} mm/h".format(row["threshold"]) for row in rows])
    axis.set_ylabel("Missed observed objects")
    axis.set_title("{} high-threshold miss sources".format(rows[0]["horizon"] if rows else "1h-2h"))
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "04_second_hour_miss_sources.png")


def main():
    args = build_parser().parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = load_values(payload, args.change_fraction)
    plot_object_counts(values, output_dir)
    plot_regime_skill(values, output_dir)
    plot_oracle_gains(values, output_dir)
    decision = build_miss_source_summary(
        values,
        args.primary_horizon,
        args.minimum_nonadvective_miss_share,
    )
    plot_miss_sources(decision, output_dir)
    decision.update(
        {
            "protocol": payload.get("protocol"),
            "samples": payload.get("samples"),
            "case_clusters": payload.get("case_clusters"),
            "change_fraction": args.change_fraction,
        }
    )
    with open(output_dir / "decision_summary.json", "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
