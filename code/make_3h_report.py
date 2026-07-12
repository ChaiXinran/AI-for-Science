import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "three_hour_comparison"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "Radar-only": {
        "metrics": ROOT / "results" / "quick_3h_radar" / "metrics.json",
        "sample": ROOT / "results" / "quick_3h_radar" / "sample_0000",
    },
    "PWV-coupled": {
        "metrics": ROOT / "results" / "quick_3h_pwv" / "metrics.json",
        "sample": ROOT / "results" / "quick_3h_pwv" / "sample_0000",
    },
    "PWV-coupled V2": {
        "metrics": ROOT / "results" / "quick_3h_pwv_v2" / "metrics.json",
        "sample": ROOT / "results" / "quick_3h_pwv_v2" / "sample_0000",
    },
    "PWV-coupled V2 Focus": {
        "metrics": ROOT / "results" / "quick_3h_pwv_v2_focus" / "metrics.json",
        "sample": ROOT / "results" / "quick_3h_pwv_v2_focus" / "sample_0000",
    },
    "PWV-coupled V2 Tuned": {
        "metrics": ROOT / "results" / "quick_3h_pwv_v2_tuned" / "metrics.json",
        "sample": ROOT / "results" / "quick_3h_pwv_v2_tuned" / "sample_0000",
    },
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics():
    required = ["Radar-only", "PWV-coupled"]
    missing = [str(EXPERIMENTS[name]["metrics"]) for name in required if not EXPERIMENTS[name]["metrics"].exists()]
    if missing:
        raise SystemExit("Missing required metrics files:\n" + "\n".join(missing))
    return {name: read_json(cfg["metrics"]) for name, cfg in EXPERIMENTS.items() if cfg["metrics"].exists()}


def lead_series(data, group, metric):
    items = data["lead_time_metrics"][group]
    x = np.array([item["lead_minutes"] / 60.0 for item in items], dtype=float)
    y = np.array([item[metric] for item in items], dtype=float)
    return x, y


def draw_horizon_band(ax):
    ax.axvspan(2.0, 3.0, color="0.92", zorder=0)
    ax.text(2.5, 0.98, "2-3h", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.35")


def save_lead_curves(metrics):
    for metric in ("mae", "rmse"):
        fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
        x_p, y_p = lead_series(metrics["Radar-only"], "persistence", metric)
        ax.plot(x_p, y_p, color="0.25", linewidth=2.4, label="Persistence")
        for name, data in metrics.items():
            x, y = lead_series(data, "model", metric)
            ax.plot(x, y, marker="o", markersize=3.5, linewidth=2.0, label=name)
        draw_horizon_band(ax)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by lead time")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / f"lead_{metric}.png")
        plt.close(fig)


def save_gap_curve(metrics):
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    _, persistence = lead_series(metrics["Radar-only"], "persistence", "mae")
    for name, data in metrics.items():
        x, y = lead_series(data, "model", "mae")
        gap = y - persistence
        ax.plot(x, gap, marker="o", markersize=3.5, linewidth=2.0, label=f"{name} - persistence")
    draw_horizon_band(ax)
    ax.axhline(0.0, color="0.25", linewidth=1.4)
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("MAE gap")
    ax.set_title("MAE gap against persistence")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "lead_mae_gap_vs_persistence.png")
    plt.close(fig)


def save_horizon_bars(metrics):
    horizons = list(metrics["Radar-only"]["horizon_metrics"]["model"].keys())
    labels = ["Persistence"] + list(metrics.keys())
    for metric in ("mae", "rmse"):
        values = {"Persistence": [metrics["Radar-only"]["horizon_metrics"]["persistence"][h][metric] for h in horizons]}
        for name, data in metrics.items():
            values[name] = [data["horizon_metrics"]["model"][h][metric] for h in horizons]
        x = np.arange(len(horizons))
        width = min(0.22, 0.8 / max(len(labels), 1))
        offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width
        fig, ax = plt.subplots(figsize=(9.4, 5.1), dpi=180)
        for offset, label in zip(offsets, labels):
            bars = ax.bar(x + offset, values[label], width, label=label)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Hourly horizon {metric.upper()}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(OUT / f"horizon_{metric}.png")
        plt.close(fig)


def save_threshold_metrics(metrics):
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
            y = [values[str(float(t))][key] for t in thresholds]
            ax.plot(thresholds, y, marker="o", linewidth=2.0, label=name)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(thresholds)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 1].set_xlabel("Threshold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_metrics.png")
    plt.close(fig)


def open_rgb(path, size=(96, 96)):
    return Image.open(path).convert("RGB").resize(size)


def colorize_gray(path, size=(96, 96), cmap="viridis", stretch=False):
    arr = np.array(Image.open(path).convert("L").resize(size)).astype("float32") / 255.0
    if stretch:
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    cm = plt.get_cmap(cmap)
    return Image.fromarray((cm(arr)[..., :3] * 255).astype("uint8"))


def save_sample_grid():
    radar = EXPERIMENTS["Radar-only"]["sample"]
    pwv = EXPERIMENTS["PWV-coupled"]["sample"]
    columns = [("t+1\n0.1h", 0), ("t+10\n1.0h", 9), ("t+20\n2.0h", 19), ("t+25\n2.5h", 24), ("t+30\n3.0h", 29)]
    rows = [
        ("Ground truth", lambda i: open_rgb(radar / f"gt_{i:02d}.png")),
        ("Persistence", lambda i: open_rgb(radar / f"ps_{i:02d}.png")),
        ("Radar-only", lambda i: open_rgb(radar / f"pd_{i:02d}.png")),
        ("PWV-coupled", lambda i: open_rgb(pwv / f"pd_{i:02d}.png")),
        ("Coupling C", lambda i: colorize_gray(pwv / f"c_{i:02d}.png", cmap="magma", stretch=False)),
    ]
    pwv_v2 = EXPERIMENTS["PWV-coupled V2"]["sample"]
    if pwv_v2.exists():
        rows.insert(4, ("PWV-coupled V2", lambda i: open_rgb(pwv_v2 / f"pd_{i:02d}.png")))
        rows.insert(6, ("Coupling C V2", lambda i: colorize_gray(pwv_v2 / f"c_{i:02d}.png", cmap="magma", stretch=False)))
    pwv_v2_focus = EXPERIMENTS["PWV-coupled V2 Focus"]["sample"]
    if pwv_v2_focus.exists():
        insert_at = 5 if pwv_v2.exists() else 4
        rows.insert(insert_at, ("PWV V2 Focus", lambda i: open_rgb(pwv_v2_focus / f"pd_{i:02d}.png")))
        rows.insert(insert_at + 3, ("C V2 Focus", lambda i: colorize_gray(pwv_v2_focus / f"c_{i:02d}.png", cmap="magma", stretch=False)))
    pwv_v2_tuned = EXPERIMENTS["PWV-coupled V2 Tuned"]["sample"]
    if pwv_v2_tuned.exists():
        insert_at = 6 if pwv_v2_focus.exists() else (5 if pwv_v2.exists() else 4)
        rows.insert(insert_at, ("PWV V2 Tuned", lambda i: open_rgb(pwv_v2_tuned / f"pd_{i:02d}.png")))
        rows.insert(insert_at + 4, ("C V2 Tuned", lambda i: colorize_gray(pwv_v2_tuned / f"c_{i:02d}.png", cmap="magma", stretch=False)))
    if (pwv / "pwv_00.png").exists():
        rows.append(("PWV input", lambda i: colorize_gray(pwv / f"pwv_{min(i, 8):02d}.png", cmap="viridis", stretch=True)))

    cell_w, cell_h = 96, 96
    label_w, top_h = 142, 42
    canvas = Image.new("RGB", (label_w + len(columns) * cell_w, top_h + len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (label, _) in enumerate(columns):
        draw.multiline_text((label_w + c * cell_w + 27, 6), label, fill="black", align="center", spacing=2)
    for r, (label, getter) in enumerate(rows):
        y = top_h + r * cell_h
        draw.text((8, y + 40), label, fill="black")
        for c, (_, idx) in enumerate(columns):
            canvas.paste(getter(idx), (label_w + c * cell_w, y))
            if label.startswith("Coupling C"):
                draw.text((label_w + c * cell_w + 4, y + 4), "C=0.50", fill="white")
    canvas.save(OUT / "sample_0000_3h_grid.png")


def save_summary(metrics):
    radar_h = metrics["Radar-only"]["horizon_metrics"]["model"]["2h-3h"]
    pwv_h = metrics["PWV-coupled"]["horizon_metrics"]["model"]["2h-3h"]
    pers_h = metrics["Radar-only"]["horizon_metrics"]["persistence"]["2h-3h"]
    summary = {
        "overall": {
            "persistence": metrics["Radar-only"]["persistence"],
            "radar_only": metrics["Radar-only"]["model"],
            "pwv_coupled": metrics["PWV-coupled"]["model"],
            "pwv_coupling_mean": metrics["PWV-coupled"].get("coupling_mean"),
        },
        "two_to_three_hour": {
            "persistence": pers_h,
            "radar_only": radar_h,
            "pwv_coupled": pwv_h,
            "radar_mae_gap_vs_persistence": radar_h["mae"] - pers_h["mae"],
            "pwv_mae_gap_vs_persistence": pwv_h["mae"] - pers_h["mae"],
        },
        "takeaways": [
            "Persistence is still best overall, especially before 2h.",
            "Radar-only becomes close to persistence in the 2h-3h bin.",
            "PWV-coupled is worse than radar-only in this quick run and its coupling mean is near 0.5.",
        ],
    }
    if "PWV-coupled V2" in metrics:
        v2_h = metrics["PWV-coupled V2"]["horizon_metrics"]["model"]["2h-3h"]
        summary["overall"]["pwv_coupled_v2"] = metrics["PWV-coupled V2"]["model"]
        summary["overall"]["pwv_coupled_v2_coupling_mean"] = metrics["PWV-coupled V2"].get("coupling_mean")
        summary["overall"]["pwv_coupled_v2_coupling_std"] = metrics["PWV-coupled V2"].get("coupling_std")
        summary["two_to_three_hour"]["pwv_coupled_v2"] = v2_h
        summary["two_to_three_hour"]["pwv_v2_mae_gap_vs_persistence"] = v2_h["mae"] - pers_h["mae"]
    if "PWV-coupled V2 Focus" in metrics:
        v2_focus_h = metrics["PWV-coupled V2 Focus"]["horizon_metrics"]["model"]["2h-3h"]
        summary["overall"]["pwv_coupled_v2_focus"] = metrics["PWV-coupled V2 Focus"]["model"]
        summary["overall"]["pwv_coupled_v2_focus_coupling_mean"] = metrics["PWV-coupled V2 Focus"].get("coupling_mean")
        summary["overall"]["pwv_coupled_v2_focus_coupling_std"] = metrics["PWV-coupled V2 Focus"].get("coupling_std")
        summary["two_to_three_hour"]["pwv_coupled_v2_focus"] = v2_focus_h
        summary["two_to_three_hour"]["pwv_v2_focus_mae_gap_vs_persistence"] = v2_focus_h["mae"] - pers_h["mae"]
    if "PWV-coupled V2 Tuned" in metrics:
        v2_tuned_h = metrics["PWV-coupled V2 Tuned"]["horizon_metrics"]["model"]["2h-3h"]
        summary["overall"]["pwv_coupled_v2_tuned"] = metrics["PWV-coupled V2 Tuned"]["model"]
        summary["overall"]["pwv_coupled_v2_tuned_coupling_mean"] = metrics["PWV-coupled V2 Tuned"].get("coupling_mean")
        summary["overall"]["pwv_coupled_v2_tuned_coupling_std"] = metrics["PWV-coupled V2 Tuned"].get("coupling_std")
        summary["two_to_three_hour"]["pwv_coupled_v2_tuned"] = v2_tuned_h
        summary["two_to_three_hour"]["pwv_v2_tuned_mae_gap_vs_persistence"] = v2_tuned_h["mae"] - pers_h["mae"]
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    metrics = load_metrics()
    save_lead_curves(metrics)
    save_gap_curve(metrics)
    save_horizon_bars(metrics)
    save_threshold_metrics(metrics)
    save_sample_grid()
    save_summary(metrics)
    print(OUT)


if __name__ == "__main__":
    main()
