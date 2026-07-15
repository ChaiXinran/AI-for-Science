import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from make_server_3h_report import (
    colorize_gray,
    open_rgb,
    read_json,
    save_cra,
    save_extreme_threshold_metrics,
    save_fss,
    save_horizon_bars,
    save_intensity_bin_improvement,
    save_intensity_bin_metrics,
    save_lead_curves,
    save_neighborhood_csi,
    save_object_metrics,
    save_pearson,
    save_psd_error,
    save_psd_plots,
    save_threshold_metrics,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Compare original PWV V3 with lead-time-conditioned PWV V3")
    parser.add_argument("--baseline_run_root", type=str, required=True)
    parser.add_argument("--new_run_root", type=str, required=True)
    parser.add_argument("--baseline_label", type=str, default="Original PWV V3")
    parser.add_argument("--new_label", type=str, default="Lead-time PWV V3")
    parser.add_argument("--output_dir", type=str, default="")
    return parser


def metric_path(run_root):
    return Path(run_root) / "results" / "pwv_v3_3h" / "metrics.json"


def sample_path(run_root):
    return Path(run_root) / "results" / "pwv_v3_3h" / "sample_0000"


def load_metrics(args):
    paths = {
        args.baseline_label: metric_path(args.baseline_run_root),
        args.new_label: metric_path(args.new_run_root),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing required metrics files:\n" + "\n".join(missing))
    return {label: read_json(path) for label, path in paths.items()}


def available_columns(sample_dir):
    preferred = [
        ("t+1\n0.1h", 0),
        ("t+10\n1.0h", 9),
        ("t+20\n2.0h", 19),
        ("t+25\n2.5h", 24),
        ("t+30\n3.0h", 29),
    ]
    return [(label, idx) for label, idx in preferred if (sample_dir / f"gt_{idx:02d}.png").exists()]


def save_sample_grid(args, out_dir):
    baseline_sample = sample_path(args.baseline_run_root)
    new_sample = sample_path(args.new_run_root)
    if not baseline_sample.exists() or not new_sample.exists():
        return

    columns = available_columns(new_sample) or available_columns(baseline_sample)
    if not columns:
        return

    rows = [
        ("Ground truth", lambda i: open_rgb(new_sample / f"gt_{i:02d}.png")),
        ("Persistence", lambda i: open_rgb(new_sample / f"ps_{i:02d}.png")),
        (args.baseline_label, lambda i: open_rgb(baseline_sample / f"pd_{i:02d}.png")),
        (args.new_label, lambda i: open_rgb(new_sample / f"pd_{i:02d}.png")),
    ]
    for label, path in ((args.baseline_label, baseline_sample), (args.new_label, new_sample)):
        if (path / "c_00.png").exists():
            rows.append((f"Coupling {label}", lambda i, p=path: colorize_gray(p / f"c_{i:02d}.png", cmap="magma")))
        if (path / "s_00.png").exists():
            rows.append((f"Support {label}", lambda i, p=path: colorize_gray(p / f"s_{i:02d}.png", cmap="magma")))
        if (path / "oc_00.png").exists():
            rows.append((f"Object center {label}", lambda i, p=path: colorize_gray(p / f"oc_{i:02d}.png", cmap="magma", stretch=True)))
        if (path / "om_00.png").exists():
            rows.append((f"Object mask {label}", lambda i, p=path: colorize_gray(p / f"om_{i:02d}.png", cmap="magma", stretch=True)))
    if (new_sample / "pwv_00.png").exists():
        rows.append(("PWV input", lambda i: colorize_gray(new_sample / f"pwv_{min(i, 8):02d}.png", cmap="viridis", stretch=True)))

    cell_w, cell_h = 96, 96
    label_w, top_h = 168, 42
    canvas = Image.new("RGB", (label_w + len(columns) * cell_w, top_h + len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (label, _) in enumerate(columns):
        draw.multiline_text((label_w + c * cell_w + 27, 6), label, fill="black", align="center", spacing=2)
    for r, (label, getter) in enumerate(rows):
        y = top_h + r * cell_h
        draw.text((8, y + 40), label, fill="black")
        for c, (_, idx) in enumerate(columns):
            try:
                canvas.paste(getter(idx), (label_w + c * cell_w, y))
            except FileNotFoundError:
                continue
    canvas.save(out_dir / "sample_0000_original_vs_leadtime_v3.png")


def scalar_delta(base, new, key):
    if base is None or new is None or key not in base or key not in new:
        return None
    return new[key] - base[key]


def summarize(metrics, args, out_dir):
    baseline = metrics[args.baseline_label]
    new = metrics[args.new_label]
    summary = {
        "labels": {
            "baseline": args.baseline_label,
            "new": args.new_label,
        },
        "paths": {
            "baseline_run_root": str(Path(args.baseline_run_root)),
            "new_run_root": str(Path(args.new_run_root)),
        },
        "overall": {
            "baseline": baseline.get("model"),
            "new": new.get("model"),
            "delta_new_minus_baseline": {
                key: scalar_delta(baseline.get("model"), new.get("model"), key)
                for key in ("mae", "rmse")
            },
        },
        "coupling": {
            "baseline_mean": baseline.get("coupling_mean"),
            "baseline_std": baseline.get("coupling_std"),
            "new_mean": new.get("coupling_mean"),
            "new_std": new.get("coupling_std"),
            "baseline_support_mean": baseline.get("support_mean"),
            "new_support_mean": new.get("support_mean"),
        },
        "horizon_delta_new_minus_baseline": {},
        "threshold_delta_new_minus_baseline": {},
        "object_metrics": {
            "baseline": baseline.get("object_metrics", {}).get("model"),
            "new": new.get("object_metrics", {}).get("model"),
        },
    }

    for horizon, base_values in baseline.get("horizon_metrics", {}).get("model", {}).items():
        new_values = new.get("horizon_metrics", {}).get("model", {}).get(horizon)
        if new_values is None:
            continue
        summary["horizon_delta_new_minus_baseline"][horizon] = {
            key: scalar_delta(base_values, new_values, key)
            for key in ("mae", "rmse")
        }

    for threshold, base_values in baseline.get("event_metrics", {}).get("model", {}).items():
        new_values = new.get("event_metrics", {}).get("model", {}).get(threshold)
        if new_values is None:
            continue
        summary["threshold_delta_new_minus_baseline"][threshold] = {
            key: scalar_delta(base_values, new_values, key)
            for key in ("csi", "pod", "far", "hss")
        }

    with open(out_dir / "summary_original_vs_leadtime_v3.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.new_run_root) / "reports" / "pwv_v3_leadtime_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args)
    save_lead_curves(metrics, out_dir)
    save_horizon_bars(metrics, out_dir)
    save_threshold_metrics(metrics, out_dir)
    save_extreme_threshold_metrics(metrics, out_dir)
    save_neighborhood_csi(metrics, out_dir)
    save_fss(metrics, out_dir)
    save_pearson(metrics, out_dir)
    save_cra(metrics, out_dir)
    save_object_metrics(metrics, out_dir)
    save_intensity_bin_metrics(metrics, out_dir)
    save_intensity_bin_improvement(metrics, out_dir)
    save_psd_plots(metrics, out_dir)
    save_psd_error(metrics, out_dir)
    save_sample_grid(args, out_dir)
    summarize(metrics, args, out_dir)
    print(f"saved PWV V3 lead-time comparison report to {out_dir}")


if __name__ == "__main__":
    main()
