import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "quick_comparison"
OUT.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = {
    "Radar-only": {
        "metrics": ROOT / "results" / "quick_exp" / "metrics.json",
        "log": ROOT / "checkpoints" / "quick_exp" / "train_log.csv",
        "sample": ROOT / "results" / "quick_exp" / "sample_0000",
    },
    "PWV-coupled": {
        "metrics": ROOT / "results" / "quick_pwv_coupled" / "metrics.json",
        "log": ROOT / "checkpoints" / "quick_pwv_coupled" / "train_log.csv",
        "sample": ROOT / "results" / "quick_pwv_coupled" / "sample_0000",
    },
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_log(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    latest_by_epoch = {}
    for row in rows:
        latest_by_epoch[int(row["epoch"])] = row
    return [latest_by_epoch[k] for k in sorted(latest_by_epoch)]


def save_metric_bars(metrics):
    labels = ["Persistence", "Radar-only", "PWV-coupled"]
    mae = [
        metrics["Radar-only"]["persistence"]["mae"],
        metrics["Radar-only"]["model"]["mae"],
        metrics["PWV-coupled"]["model"]["mae"],
    ]
    rmse = [
        metrics["Radar-only"]["persistence"]["rmse"],
        metrics["Radar-only"]["model"]["rmse"],
        metrics["PWV-coupled"]["model"]["rmse"],
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    bars1 = ax.bar(x - width / 2, mae, width, label="MAE")
    bars2 = ax.bar(x + width / 2, rmse, width, label="RMSE")
    ax.set_ylabel("Error, lower is better")
    ax.set_title("Quick experiment error summary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "error_summary.png")
    plt.close(fig)


def save_threshold_lines(metrics):
    thresholds = metrics["Radar-only"]["thresholds"]
    series = {
        "Persistence": metrics["Radar-only"]["event_metrics"]["persistence"],
        "Radar-only": metrics["Radar-only"]["event_metrics"]["model"],
        "PWV-coupled": metrics["PWV-coupled"]["event_metrics"]["model"],
    }
    plot_metrics = [("csi", "CSI, higher is better"), ("pod", "POD, higher is better"),
                    ("far", "FAR, lower is better"), ("hss", "HSS, higher is better")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=180, sharex=True)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, plot_metrics):
        for name, values in series.items():
            y = [values[str(float(t))][key] for t in thresholds]
            ax.plot(thresholds, y, marker="o", linewidth=2, label=name)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(thresholds)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.25)
        if key in ("csi", "pod", "far", "hss"):
            ax.set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Threshold")
    axes[3].set_xlabel("Threshold")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Threshold skill comparison")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_metrics.png")
    plt.close(fig)


def save_training_curves(logs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=180)
    for name, rows in logs.items():
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["val_weighted_l1"] for r in rows], marker="o", linewidth=2, label=name)
        axes[1].plot(epochs, [r["g_total"] for r in rows], marker="o", linewidth=2, label=name)
    axes[0].set_title("Validation weighted L1")
    axes[1].set_title("Generator total loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Quick training curves")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png")
    plt.close(fig)


def open_resize(path, size=(96, 96)):
    return Image.open(path).convert("RGB").resize(size)


def colorize_gray(path, size=(96, 96), cmap="viridis", stretch=False):
    arr = np.array(Image.open(path).convert("L").resize(size)).astype("float32") / 255.0
    if stretch:
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    cm = plt.get_cmap(cmap)
    rgb = (cm(arr)[..., :3] * 255).astype("uint8")
    return Image.fromarray(rgb)


def save_sample_grid():
    radar_sample = EXPERIMENTS["Radar-only"]["sample"]
    pwv_sample = EXPERIMENTS["PWV-coupled"]["sample"]
    cols = [("t+1", 0), ("t+6", 5), ("t+11", 10), ("t+20", 19)]
    rows = [
        ("Ground truth", lambda idx: open_resize(radar_sample / f"gt_{idx:02d}.png")),
        ("Persistence", lambda idx: open_resize(radar_sample / f"ps_{idx:02d}.png")),
        ("Radar-only", lambda idx: open_resize(radar_sample / f"pd_{idx:02d}.png")),
        ("PWV-coupled", lambda idx: open_resize(pwv_sample / f"pd_{idx:02d}.png")),
        ("Coupling C", lambda idx: colorize_gray(pwv_sample / f"c_{idx:02d}.png", cmap="magma")),
    ]
    if (pwv_sample / "pwv_00.png").exists():
        rows.append(("PWV input", lambda idx: colorize_gray(pwv_sample / f"pwv_{min(idx, 8):02d}.png", cmap="viridis", stretch=True)))

    cell_w, cell_h = 96, 96
    label_w, top_h = 138, 32
    canvas = Image.new("RGB", (label_w + len(cols) * cell_w, top_h + len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (label, _) in enumerate(cols):
        draw.text((label_w + c * cell_w + 30, 10), label, fill="black")
    for r, (label, getter) in enumerate(rows):
        y = top_h + r * cell_h
        draw.text((8, y + 40), label, fill="black")
        for c, (_, idx) in enumerate(cols):
            cell = getter(idx)
            canvas.paste(cell, (label_w + c * cell_w, y))
            if label == "Coupling C":
                draw.text((label_w + c * cell_w + 4, y + 4), "~0.50", fill="white")
    canvas.save(OUT / "sample_0000_grid.png")


def save_summary_json(metrics):
    summary = {
        "takeaways": [
            "Persistence remains strongest in this quick run.",
            "PWV-coupled has lower FAR than radar-only at thresholds 1, 5, 10, and 20, but lower POD and CSI.",
            "Coupling field is still near 0.5, so the PWV branch has not yet learned a spatially selective C_t in the short run.",
        ],
        "error": {
            "persistence_mae": metrics["Radar-only"]["persistence"]["mae"],
            "radar_only_mae": metrics["Radar-only"]["model"]["mae"],
            "pwv_coupled_mae": metrics["PWV-coupled"]["model"]["mae"],
            "persistence_rmse": metrics["Radar-only"]["persistence"]["rmse"],
            "radar_only_rmse": metrics["Radar-only"]["model"]["rmse"],
            "pwv_coupled_rmse": metrics["PWV-coupled"]["model"]["rmse"],
        },
        "pwv_coupling_mean": metrics["PWV-coupled"].get("coupling_mean"),
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    metrics = {name: read_json(cfg["metrics"]) for name, cfg in EXPERIMENTS.items()}
    logs = {name: read_log(cfg["log"]) for name, cfg in EXPERIMENTS.items()}
    save_metric_bars(metrics)
    save_threshold_lines(metrics)
    save_training_curves(logs)
    save_sample_grid()
    save_summary_json(metrics)
    print(OUT)


if __name__ == "__main__":
    main()
