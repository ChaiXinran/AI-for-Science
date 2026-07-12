import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


EXPERIMENTS = {
    "Radar-only": {
        "metrics": "results/radar_3h/metrics.json",
        "sample": "results/radar_3h/sample_0000",
    },
    "PWV V2": {
        "metrics": "results/pwv_v2_3h/metrics.json",
        "sample": "results/pwv_v2_3h/sample_0000",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="Build comparison plots for server 3h experiments")
    parser.add_argument("--run_root", type=str, default="/root/autodl-tmp/nowcastnet_runs/north_china_3h")
    parser.add_argument("--output_dir", type=str, default="")
    return parser


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(run_root):
    metrics = {}
    missing = []
    for name, cfg in EXPERIMENTS.items():
        path = run_root / cfg["metrics"]
        if path.exists():
            metrics[name] = read_json(path)
        else:
            missing.append(str(path))
    if missing:
        raise SystemExit("Missing metrics files:\n" + "\n".join(missing))
    return metrics


def lead_series(data, group, metric):
    items = data["lead_time_metrics"][group]
    x = np.array([item["lead_minutes"] / 60.0 for item in items], dtype=float)
    y = np.array([item[metric] for item in items], dtype=float)
    return x, y


def threshold_key(value):
    return str(float(value))


def draw_horizon_band(ax):
    ax.axvspan(2.0, 3.0, color="0.92", zorder=0)
    ax.text(
        2.5,
        0.98,
        "2-3h",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color="0.35",
    )


def save_lead_curves(metrics, out_dir):
    for metric in ("mae", "rmse"):
        fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=180)
        x_p, y_p = lead_series(metrics["Radar-only"], "persistence", metric)
        ax.plot(x_p, y_p, color="0.25", linewidth=2.4, label="Persistence")
        for name, data in metrics.items():
            x, y = lead_series(data, "model", metric)
            ax.plot(x, y, marker="o", markersize=3.2, linewidth=2.0, label=name)
        draw_horizon_band(ax)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by lead time")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"lead_{metric}.png")
        plt.close(fig)


def save_horizon_bars(metrics, out_dir):
    horizons = list(metrics["Radar-only"]["horizon_metrics"]["model"].keys())
    labels = ["Persistence"] + list(metrics.keys())
    for metric in ("mae", "rmse"):
        values = {
            "Persistence": [
                metrics["Radar-only"]["horizon_metrics"]["persistence"][h][metric]
                for h in horizons
            ]
        }
        for name, data in metrics.items():
            values[name] = [data["horizon_metrics"]["model"][h][metric] for h in horizons]
        x = np.arange(len(horizons))
        width = min(0.24, 0.8 / max(len(labels), 1))
        offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width
        fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=180)
        for offset, label in zip(offsets, labels):
            bars = ax.bar(x + offset, values[label], width, label=label)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Hourly horizon {metric.upper()}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"horizon_{metric}.png")
        plt.close(fig)


def save_threshold_metrics(metrics, out_dir):
    thresholds = metrics["Radar-only"]["thresholds"]
    series = {"Persistence": metrics["Radar-only"]["event_metrics"]["persistence"]}
    for name, data in metrics.items():
        series[name] = data["event_metrics"]["model"]

    panels = [
        ("csi", "CSI, higher is better"),
        ("pod", "POD, higher is better"),
        ("far", "FAR, lower is better"),
        ("hss", "HSS, higher is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=180, sharex=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name, values in series.items():
            y = [values[threshold_key(t)][key] for t in thresholds]
            ax.plot(thresholds, y, marker="o", linewidth=2.0, label=name)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(thresholds)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if key != "far":
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 1].set_xlabel("Threshold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_metrics.png")
    plt.close(fig)


def open_rgb(path, size=(96, 96)):
    return Image.open(path).convert("RGB").resize(size)


def colorize_gray(path, size=(96, 96), cmap="magma", stretch=False):
    arr = np.array(Image.open(path).convert("L").resize(size)).astype("float32") / 255.0
    if stretch:
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    cm = plt.get_cmap(cmap)
    return Image.fromarray((cm(arr)[..., :3] * 255).astype("uint8"))


def save_sample_grid(run_root, out_dir):
    radar = run_root / EXPERIMENTS["Radar-only"]["sample"]
    pwv = run_root / EXPERIMENTS["PWV V2"]["sample"]
    if not radar.exists() or not pwv.exists():
        return

    columns = [
        ("t+1\n0.1h", 0),
        ("t+10\n1.0h", 9),
        ("t+20\n2.0h", 19),
        ("t+25\n2.5h", 24),
        ("t+30\n3.0h", 29),
    ]
    rows = [
        ("Ground truth", lambda i: open_rgb(radar / f"gt_{i:02d}.png")),
        ("Persistence", lambda i: open_rgb(radar / f"ps_{i:02d}.png")),
        ("Radar-only", lambda i: open_rgb(radar / f"pd_{i:02d}.png")),
        ("PWV V2", lambda i: open_rgb(pwv / f"pd_{i:02d}.png")),
        ("Coupling C", lambda i: colorize_gray(pwv / f"c_{i:02d}.png", cmap="magma")),
    ]
    if (pwv / "pwv_00.png").exists():
        rows.append(("PWV input", lambda i: colorize_gray(pwv / f"pwv_{min(i, 8):02d}.png", cmap="viridis", stretch=True)))

    cell_w, cell_h = 96, 96
    label_w, top_h = 128, 42
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
    canvas.save(out_dir / "sample_0000_3h_grid.png")


def summarize(metrics, out_dir):
    radar = metrics["Radar-only"]
    pwv = metrics["PWV V2"]
    summary = {
        "overall": {
            "persistence": radar["persistence"],
            "radar_only": radar["model"],
            "pwv_v2": pwv["model"],
            "pwv_coupling_mean": pwv.get("coupling_mean"),
            "pwv_coupling_std": pwv.get("coupling_std"),
        },
        "horizon_metrics": {
            "persistence": radar["horizon_metrics"]["persistence"],
            "radar_only": radar["horizon_metrics"]["model"],
            "pwv_v2": pwv["horizon_metrics"]["model"],
        },
        "event_metrics": {
            "persistence": radar["event_metrics"]["persistence"],
            "radar_only": radar["event_metrics"]["model"],
            "pwv_v2": pwv["event_metrics"]["model"],
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    args = build_parser().parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.output_dir) if args.output_dir else run_root / "reports" / "comparison_3h"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(run_root)
    save_lead_curves(metrics, out_dir)
    save_horizon_bars(metrics, out_dir)
    save_threshold_metrics(metrics, out_dir)
    save_sample_grid(run_root, out_dir)
    summarize(metrics, out_dir)
    print(f"saved server comparison report to {out_dir}")


if __name__ == "__main__":
    main()
