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
        "required": False,
    },
    "PWV V2": {
        "metrics": "results/pwv_v2_3h/metrics.json",
        "sample": "results/pwv_v2_3h/sample_0000",
        "required": False,
    },
    "PWV V3": {
        "metrics": "results/pwv_v3_3h/metrics.json",
        "sample": "results/pwv_v3_3h/sample_0000",
        "required": False,
    },
    "PWV V4": {
        "metrics": "results/pwv_v4_3h/metrics.json",
        "sample": "results/pwv_v4_3h/sample_0000",
        "required": False,
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="Build comparison plots for server 3h experiments")
    parser.add_argument("--run_root", type=str, default="/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical")
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
        elif cfg.get("required", True):
            missing.append(str(path))
    if missing:
        raise SystemExit("Missing metrics files:\n" + "\n".join(missing))
    if not metrics:
        expected = [str(run_root / cfg["metrics"]) for cfg in EXPERIMENTS.values()]
        raise SystemExit("No metrics files found. Expected one or more of:\n" + "\n".join(expected))
    return metrics


def reference_name(metrics):
    if "Radar-only" in metrics:
        return "Radar-only"
    return next(iter(metrics.keys()))


def reference_data(metrics):
    return metrics[reference_name(metrics)]


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
        x_p, y_p = lead_series(reference_data(metrics), "persistence", metric)
        ax.plot(x_p, y_p, color="0.25", linewidth=2.4, label="Persistence")
        for name, data in metrics.items():
            x, y = lead_series(data, "model", metric)
            ax.plot(x, y, marker="o", markersize=3.2, linewidth=2.0, label=name)
        draw_horizon_band(ax)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(f"{metric.upper()} (mm/h)")
        ax.set_title(f"{metric.upper()} by lead time")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"lead_{metric}.png")
        plt.close(fig)


def save_horizon_bars(metrics, out_dir):
    ref = reference_data(metrics)
    horizons = list(ref["horizon_metrics"]["model"].keys())
    labels = ["Persistence"] + list(metrics.keys())
    for metric in ("mae", "rmse"):
        values = {
            "Persistence": [
                ref["horizon_metrics"]["persistence"][h][metric]
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
        ax.set_ylabel(f"{metric.upper()} (mm/h)")
        ax.set_title(f"Hourly horizon {metric.upper()}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"horizon_{metric}.png")
        plt.close(fig)


def save_threshold_metrics(metrics, out_dir):
    ref = reference_data(metrics)
    thresholds = ref["thresholds"]
    series = {"Persistence": ref["event_metrics"]["persistence"]}
    for name, data in metrics.items():
        series[name] = data["event_metrics"]["model"]

    panels = [
        ("csi", "CSI, higher is better"),
        ("pod", "POD, higher is better"),
        ("far", "FAR, lower is better"),
        ("hss", "HSS, higher is better"),
        ("f1", "F1, higher is better"),
        ("ets", "ETS, higher is better"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), dpi=180, sharex=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name, values in series.items():
            y = [values[threshold_key(t)].get(key, np.nan) for t in thresholds]
            ax.plot(thresholds, y, marker="o", linewidth=2.0, label=name)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(thresholds)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if key != "far":
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel("Rain-rate threshold (mm/h)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_metrics.png")
    plt.close(fig)


def save_pearson(metrics, out_dir):
    if not any("pearson" in data for data in metrics.values()):
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=180)
    ref = reference_data(metrics)
    if "pearson" in ref and "persistence" in ref["pearson"]:
        items = ref["pearson"]["persistence"].get("lead_time", [])
        if items:
            x = [item["lead_minutes"] / 60.0 for item in items]
            y = [item["pearson"] for item in items]
            ax.plot(x, y, color="0.25", linewidth=2.4, label="Persistence")
    for name, data in metrics.items():
        items = data.get("pearson", {}).get("model", {}).get("lead_time", [])
        if not items:
            continue
        x = [item["lead_minutes"] / 60.0 for item in items]
        y = [item["pearson"] for item in items]
        ax.plot(x, y, marker="o", linewidth=2.0, label=name)
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("Pearson spatial correlation")
    ax.set_title("Spatial correlation by lead time")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "lead_pearson.png")
    plt.close(fig)


def save_cra(metrics, out_dir):
    if not any("cra" in data for data in metrics.values()):
        return
    ref = reference_data(metrics)
    cra_ref = ref.get("cra", {}).get("model", {})
    if not cra_ref:
        return
    lead_keys = list(cra_ref.keys())
    threshold_keys = list(next(iter(cra_ref.values())).keys())
    for threshold in threshold_keys:
        labels = ["Persistence"] + list(metrics.keys())
        x = np.arange(len(lead_keys))
        width = min(0.24, 0.8 / max(len(labels), 1))
        offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), dpi=180, sharex=True)
        panels = [
            ("distance_km", "CRA displacement distance (km)"),
            ("rmse_displacement", "CRA displacement RMSE"),
            ("rmse_volume", "CRA volume RMSE"),
            ("rmse_pattern", "CRA pattern RMSE"),
        ]
        values_by_label = {}
        ref_persistence = ref.get("cra", {}).get("persistence", {})
        values_by_label["Persistence"] = ref_persistence
        for name, data in metrics.items():
            values_by_label[name] = data.get("cra", {}).get("model", {})
        for ax, (metric_key, title) in zip(axes.ravel(), panels):
            for offset, label in zip(offsets, labels):
                series = values_by_label.get(label, {})
                y = []
                for lead in lead_keys:
                    item = series.get(lead, {}).get(threshold, {}).get("metrics", {}).get(metric_key, {})
                    y.append(item.get("median", np.nan))
                ax.bar(x + offset, y, width, label=label)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xticks(x)
            ax.set_xticklabels([f"T+{int(lead) // 60}h" for lead in lead_keys])
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        suffix = threshold.replace(".", "p")
        fig.savefig(out_dir / f"cra_threshold_{suffix}.png")
        plt.close(fig)


def nested_value(data, path, default=np.nan):
    value = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def object_series_for_threshold(metrics, threshold):
    ref = reference_data(metrics)
    series = {}
    persistence = ref.get("object_metrics", {}).get("persistence", {}).get(threshold)
    if persistence:
        series["Persistence"] = persistence
    for name, data in metrics.items():
        values = data.get("object_metrics", {}).get("model", {}).get(threshold)
        if values:
            series[name] = values
    return series


def save_object_metrics(metrics, out_dir):
    if not any("object_metrics" in data for data in metrics.values()):
        return
    ref = reference_data(metrics)
    object_ref = ref.get("object_metrics", {}).get("model", {})
    if not object_ref:
        return

    panels = [
        (("object_csi",), "Object CSI, higher is better", "score"),
        (("object_pod",), "Object POD, higher is better", "score"),
        (("object_far",), "Object FAR, lower is better", "score"),
        (("object_bias",), "Object count bias", "ratio"),
        (("matched", "centroid_distance_km", "median"), "Matched centroid distance", "km"),
        (("matched", "iou", "median"), "Matched IoU, higher is better", "IoU"),
    ]

    for threshold in object_ref.keys():
        series = object_series_for_threshold(metrics, threshold)
        if not series:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), dpi=180, sharex=True)
        for ax, (path, title, ylabel) in zip(axes.ravel(), panels):
            for name, values in series.items():
                lead_items = values.get("lead_time", [])
                if not lead_items:
                    continue
                x = [item["lead_minutes"] / 60.0 for item in lead_items]
                y = [nested_value(item, path) for item in lead_items]
                ax.plot(x, y, marker="o", markersize=3.2, linewidth=2.0, label=name)
            draw_horizon_band(ax)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if path[-1] in ("object_csi", "object_pod", "object_far") or path == ("matched", "iou", "median"):
                ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xlabel("Lead time (hours)")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Object-based metrics at {float(threshold):g} mm/h", y=1.02)
        fig.tight_layout()
        suffix = threshold.replace(".", "p")
        fig.savefig(out_dir / f"object_metrics_threshold_{suffix}.png")
        plt.close(fig)


def save_extreme_threshold_metrics(metrics, out_dir):
    ref = reference_data(metrics)
    if "extreme_event_metrics" not in ref:
        return
    labels = list(ref["extreme_event_metrics"]["model"].keys())
    if not labels:
        return
    series = {"Persistence": ref["extreme_event_metrics"]["persistence"]}
    for name, data in metrics.items():
        if "extreme_event_metrics" in data:
            series[name] = data["extreme_event_metrics"]["model"]

    panels = [
        ("csi", "Extreme CSI, higher is better"),
        ("pod", "Extreme POD, higher is better"),
        ("far", "Extreme FAR, lower is better"),
        ("ets", "Extreme ETS, higher is better"),
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=180, sharex=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name, values in series.items():
            y = [values[label][key] for label in labels]
            ax.plot(x, y, marker="o", linewidth=2.0, label=name)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if key != "far":
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Target rainy-pixel quantile")
    axes[1, 1].set_xlabel("Target rainy-pixel quantile")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "extreme_threshold_metrics.png")
    plt.close(fig)


def save_neighborhood_csi(metrics, out_dir):
    ref = reference_data(metrics)
    if "neighborhood_event_metrics" not in ref:
        return
    thresholds = ref["neighborhood_thresholds"]
    series = {"Persistence": ref["neighborhood_event_metrics"]["persistence"]}
    for name, data in metrics.items():
        series[name] = data["neighborhood_event_metrics"]["model"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    x = np.arange(len(thresholds))
    width = min(0.24, 0.8 / max(len(series), 1))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    for offset, (name, values) in zip(offsets, series.items()):
        y = [values[threshold_key(t)]["csin"] for t in thresholds]
        bars = ax.bar(x + offset, y, width, label=name)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t)}" if float(t).is_integer() else str(t) for t in thresholds])
    ax.set_xlabel("Rain-rate threshold (mm/h)")
    ax.set_ylabel("5x5 neighbourhood CSI")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Paper-style CSIN")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "neighborhood_csi.png")
    plt.close(fig)


def save_fss(metrics, out_dir):
    ref = reference_data(metrics)
    if "fss" not in ref:
        return
    threshold_data = ref["fss"]["model"].get("thresholds", {})
    if not threshold_data:
        return
    labels = list(threshold_data.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.0 * len(labels), 4.6), dpi=180, squeeze=False)
    for ax, label in zip(axes.ravel(), labels):
        ref_neighborhoods = threshold_data[label]["neighborhoods"]
        sizes = [int(size) for size in ref_neighborhoods.keys()]
        series = {"Persistence": ref["fss"]["persistence"]["thresholds"][label]["neighborhoods"]}
        for name, data in metrics.items():
            if "fss" in data and label in data["fss"]["model"].get("thresholds", {}):
                series[name] = data["fss"]["model"]["thresholds"][label]["neighborhoods"]
        for name, values in series.items():
            y = [values[str(size)]["fss"] for size in sizes]
            ax.plot(sizes, y, marker="o", linewidth=2.0, label=name)
        ax.set_title(f"FSS at {label}")
        ax.set_xlabel("Neighbourhood size (pixels)")
        ax.set_ylabel("FSS")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fss_extreme.png")
    plt.close(fig)


def save_intensity_bin_metrics(metrics, out_dir):
    ref = reference_data(metrics)
    if "intensity_bin_metrics" not in ref:
        return
    bins = list(ref["intensity_bin_metrics"]["model"].keys())
    if not bins:
        return
    labels = ["Persistence"] + list(metrics.keys())
    for metric in ("mae", "rmse", "bias"):
        values = {
            "Persistence": [
                ref["intensity_bin_metrics"]["persistence"][label][metric]
                for label in bins
            ]
        }
        for name, data in metrics.items():
            if "intensity_bin_metrics" in data:
                values[name] = [data["intensity_bin_metrics"]["model"][label][metric] for label in bins]
        x = np.arange(len(bins))
        width = min(0.22, 0.8 / max(len(labels), 1))
        offsets = (np.arange(len(values)) - (len(values) - 1) / 2.0) * width
        fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=180)
        for offset, label in zip(offsets, values.keys()):
            ax.bar(x + offset, values[label], width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=25, ha="right")
        ax.set_ylabel(f"{metric.upper()} (mm/h)")
        ax.set_title(f"{metric.upper()} by observed rain-rate bin")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"intensity_bin_{metric}.png")
        plt.close(fig)


def save_intensity_bin_improvement(metrics, out_dir):
    if "Radar-only" not in metrics:
        return
    radar = metrics["Radar-only"]
    if "intensity_bin_metrics" not in radar:
        return
    bins = list(radar["intensity_bin_metrics"]["model"].keys())
    candidates = {
        name: data
        for name, data in metrics.items()
        if name != "Radar-only" and "intensity_bin_metrics" in data
    }
    if not candidates:
        return
    x = np.arange(len(bins))
    width = min(0.26, 0.8 / max(len(candidates), 1))
    offsets = (np.arange(len(candidates)) - (len(candidates) - 1) / 2.0) * width
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), dpi=180, sharex=True)
    for ax, metric in zip(axes, ("mae", "rmse")):
        for offset, (name, data) in zip(offsets, candidates.items()):
            improvement = []
            for label in bins:
                base = radar["intensity_bin_metrics"]["model"][label][metric]
                value = data["intensity_bin_metrics"]["model"][label][metric]
                improvement.append((base - value) / base * 100.0 if base else 0.0)
            ax.bar(x + offset, improvement, width, label=name)
        ax.axhline(0.0, color="0.25", linewidth=1.0)
        ax.set_ylabel(f"{metric.upper()} improvement (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(bins, rotation=25, ha="right")
    axes[0].set_title("Improvement relative to Radar-only by observed rain-rate bin")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_bin_improvement.png")
    plt.close(fig)


def save_psd_plots(metrics, out_dir):
    ref = reference_data(metrics)
    if "psd" not in ref:
        return
    wavelengths = ref["psd"]["wavelengths"]
    lead_keys = list(ref["psd"]["lead_minutes"].keys())
    for lead in lead_keys:
        fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=180)
        target = ref["psd"]["lead_minutes"][lead]["target"]
        ax.plot(wavelengths, target, color="0.15", linewidth=2.6, label="Ground truth")
        persistence = ref["psd"]["lead_minutes"][lead]["persistence"]
        ax.plot(wavelengths, persistence, color="0.45", linewidth=2.0, linestyle="--", label="Persistence")
        for name, data in metrics.items():
            values = data["psd"]["lead_minutes"][lead]["model"]
            ax.plot(wavelengths, values, marker="o", linewidth=2.0, label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(wavelengths)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Wavelength (km)")
        ax.set_ylabel("Power spectral density")
        ax.set_title(f"PSD at T+{int(lead) // 60}h")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"psd_t{lead}min.png")
        plt.close(fig)


def save_psd_error(metrics, out_dir):
    ref = reference_data(metrics)
    if "psd" not in ref:
        return
    lead_keys = list(ref["psd"]["lead_minutes"].keys())
    labels = ["Persistence"] + list(metrics.keys())
    x = np.arange(len(lead_keys))
    width = min(0.24, 0.8 / max(len(labels), 1))
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180)
    values = {
        "Persistence": [
            float(np.mean(ref["psd"]["lead_minutes"][lead]["persistence_log_rmse"]))
            for lead in lead_keys
        ]
    }
    for name, data in metrics.items():
        values[name] = [
            float(np.mean(data["psd"]["lead_minutes"][lead]["model_log_rmse"]))
            for lead in lead_keys
        ]
    for offset, label in zip(offsets, labels):
        ax.bar(x + offset, values[label], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T+{int(lead) // 60}h" for lead in lead_keys])
    ax.set_ylabel("Mean log-PSD RMSE")
    ax.set_title("PSD distance to ground truth, lower is better")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "psd_log_rmse.png")
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
    sample_paths = {name: run_root / cfg["sample"] for name, cfg in EXPERIMENTS.items()}
    existing = {name: path for name, path in sample_paths.items() if path.exists()}
    if not existing:
        return
    reference_sample = existing.get("Radar-only") or next(iter(existing.values()))

    columns = [
        ("t+1\n0.1h", 0),
        ("t+10\n1.0h", 9),
        ("t+20\n2.0h", 19),
        ("t+25\n2.5h", 24),
        ("t+30\n3.0h", 29),
    ]
    rows = [
        ("Ground truth", lambda i: open_rgb(reference_sample / f"gt_{i:02d}.png")),
        ("Persistence", lambda i: open_rgb(reference_sample / f"ps_{i:02d}.png")),
    ]
    for name, path in existing.items():
        rows.append((name, lambda i, p=path: open_rgb(p / f"pd_{i:02d}.png")))
        if (path / "c_00.png").exists():
            rows.append((f"Coupling {name}", lambda i, p=path: colorize_gray(p / f"c_{i:02d}.png", cmap="magma")))
        if (path / "s_00.png").exists():
            rows.append((f"Support {name}", lambda i, p=path: colorize_gray(p / f"s_{i:02d}.png", cmap="magma")))
        if (path / "a_00.png").exists():
            rows.append((f"Attention {name}", lambda i, p=path: colorize_gray(p / f"a_{min(i, 8):02d}.png", cmap="viridis", stretch=True)))
    pwv_sample = existing.get("PWV V2") or existing.get("PWV V3") or existing.get("PWV V4")
    if pwv_sample is not None and (pwv_sample / "pwv_00.png").exists():
        rows.append(("PWV input", lambda i, p=pwv_sample: colorize_gray(p / f"pwv_{min(i, 8):02d}.png", cmap="viridis", stretch=True)))

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
    ref = reference_data(metrics)
    radar = metrics.get("Radar-only", ref)
    pwv = metrics.get("PWV V2", {})
    summary = {
        "overall": {
            "persistence": ref["persistence"],
            "radar_only": metrics.get("Radar-only", {}).get("model"),
            "pwv_v2": pwv.get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("model"),
            "pwv_coupling_mean": pwv.get("coupling_mean"),
            "pwv_coupling_std": pwv.get("coupling_std"),
            "pwv_v3_coupling_mean": metrics.get("PWV V3", {}).get("coupling_mean"),
            "pwv_v3_support_mean": metrics.get("PWV V3", {}).get("support_mean"),
            "pwv_v4_coupling_mean": metrics.get("PWV V4", {}).get("coupling_mean"),
            "pwv_v4_support_mean": metrics.get("PWV V4", {}).get("support_mean"),
            "pwv_v4_temporal_attention_mean": metrics.get("PWV V4", {}).get("pwv_temporal_attention_mean"),
        },
        "horizon_metrics": {
            "persistence": ref["horizon_metrics"]["persistence"],
            "radar_only": metrics.get("Radar-only", {}).get("horizon_metrics", {}).get("model"),
            "pwv_v2": pwv.get("horizon_metrics", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("horizon_metrics", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("horizon_metrics", {}).get("model"),
        },
        "event_metrics": {
            "persistence": ref["event_metrics"]["persistence"],
            "radar_only": metrics.get("Radar-only", {}).get("event_metrics", {}).get("model"),
            "pwv_v2": pwv.get("event_metrics", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("event_metrics", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("event_metrics", {}).get("model"),
        },
        "extreme_thresholds": ref.get("extreme_thresholds"),
        "extreme_event_metrics": {
            "persistence": ref.get("extreme_event_metrics", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("extreme_event_metrics", {}).get("model"),
            "pwv_v2": pwv.get("extreme_event_metrics", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("extreme_event_metrics", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("extreme_event_metrics", {}).get("model"),
        },
        "intensity_bin_metrics": {
            "persistence": ref.get("intensity_bin_metrics", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("intensity_bin_metrics", {}).get("model"),
            "pwv_v2": pwv.get("intensity_bin_metrics", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("intensity_bin_metrics", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("intensity_bin_metrics", {}).get("model"),
        },
        "neighborhood_event_metrics": {
            "persistence": ref.get("neighborhood_event_metrics", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("neighborhood_event_metrics", {}).get("model"),
            "pwv_v2": pwv.get("neighborhood_event_metrics", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("neighborhood_event_metrics", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("neighborhood_event_metrics", {}).get("model"),
        },
        "fss": {
            "persistence": ref.get("fss", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("fss", {}).get("model"),
            "pwv_v2": pwv.get("fss", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("fss", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("fss", {}).get("model"),
        },
        "neighborhood_score": {
            "persistence": ref.get("neighborhood_score", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("neighborhood_score", {}).get("model"),
            "pwv_v2": pwv.get("neighborhood_score", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("neighborhood_score", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("neighborhood_score", {}).get("model"),
        },
        "pearson": {
            "persistence": ref.get("pearson", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("pearson", {}).get("model"),
            "pwv_v2": pwv.get("pearson", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("pearson", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("pearson", {}).get("model"),
        },
        "cra": {
            "persistence": ref.get("cra", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("cra", {}).get("model"),
            "pwv_v2": pwv.get("cra", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("cra", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("cra", {}).get("model"),
        },
        "object_metrics": {
            "persistence": ref.get("object_metrics", {}).get("persistence"),
            "radar_only": metrics.get("Radar-only", {}).get("object_metrics", {}).get("model"),
            "pwv_v2": pwv.get("object_metrics", {}).get("model"),
            "pwv_v3": metrics.get("PWV V3", {}).get("object_metrics", {}).get("model"),
            "pwv_v4": metrics.get("PWV V4", {}).get("object_metrics", {}).get("model"),
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
    save_sample_grid(run_root, out_dir)
    summarize(metrics, out_dir)
    print(f"saved server comparison report to {out_dir}")


if __name__ == "__main__":
    main()
