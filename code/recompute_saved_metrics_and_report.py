import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_server_3h_report import (
    colorize_gray,
    open_rgb,
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
from test_custom import (
    average_neighborhood_score,
    build_intensity_bins,
    finalize_event_metrics,
    finalize_fss_metrics,
    finalize_horizon_metrics,
    finalize_intensity_bin_metrics,
    finalize_labeled_event_metrics,
    finalize_lead_metrics,
    finalize_neighborhood_metrics,
    finalize_pearson_totals,
    finalize_psd_metrics,
    finalize_scalar_totals,
    init_cra_store,
    init_event_counts,
    init_eventwise_store,
    init_fss_totals,
    init_horizon_totals,
    init_intensity_bin_totals,
    init_labeled_event_counts,
    init_lead_totals,
    init_object_store,
    init_pearson_totals,
    init_psd_totals,
    init_scalar_totals,
    parse_float_list,
    parse_horizon_bins,
    parse_int_list,
    parse_thresholds,
    quantile_label,
    threshold_items_from_quantiles,
    summarize_cra_store,
    summarize_eventwise_store,
    summarize_object_store,
    update_cra_store,
    update_event_counts,
    update_eventwise_store,
    update_fss_totals,
    update_intensity_bin_totals,
    update_labeled_event_counts,
    update_lead_and_horizon,
    update_neighborhood_event_counts,
    update_object_store,
    update_pearson_totals,
    update_psd_totals,
    update_scalar_totals,
)


RESULT_LABELS = {
    "radar_3h": "Radar-only",
    "pwv_v2_3h": "PWV V2",
    "pwv_v3_3h": "PWV V3",
    "pwv_v4_3h": "PWV V4",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Recompute metrics and comparison plots from saved NowcastNet PNG result samples."
    )
    parser.add_argument("--runs_root", type=str, default="/root/autodl-tmp/nowcastnet_runs")
    parser.add_argument(
        "--run_dirs",
        type=str,
        default="",
        help="Comma-separated run folders under runs_root. Empty means auto-discover all folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Comparison report folder. Default: runs_root/recomputed_reports/comparison_3h",
    )
    parser.add_argument("--metrics_name", type=str, default="metrics_recomputed.json")
    parser.add_argument("--manifest_name", type=str, default="recomputed_manifest.csv")
    parser.add_argument("--intensity_scale", type=float, default=35.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--metric_thresholds", type=str, default="0.5,2,5,10,30")
    parser.add_argument("--neighborhood_metric_thresholds", type=str, default="16,32")
    parser.add_argument("--neighborhood_size", type=int, default=5)
    parser.add_argument("--extreme_quantiles", type=str, default="0.9,0.95,0.99")
    parser.add_argument("--extreme_rain_min", type=float, default=0.1)
    parser.add_argument("--intensity_bin_quantiles", type=str, default="0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--fss_quantiles", type=str, default="0.95,0.99")
    parser.add_argument("--fss_neighborhood_sizes", type=str, default="1,3,5,9,15")
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--horizon_bins", type=str, default="0-1,1-2,2-3")
    parser.add_argument("--psd_lead_minutes", type=str, default="60,120,180")
    parser.add_argument("--psd_wavelengths", type=str, default="4,8,16,32,64")
    parser.add_argument("--grid_km", type=float, default=1.0)
    parser.add_argument("--cra_thresholds", type=str, default="16")
    parser.add_argument("--cra_lead_minutes", type=str, default="60,120,180")
    parser.add_argument("--cra_max_shift", type=int, default=12)
    parser.add_argument("--object_thresholds", type=str, default="16")
    parser.add_argument("--object_min_area", type=int, default=4)
    parser.add_argument("--object_iou_threshold", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--only_with_metrics",
        action="store_true",
        help="Only use result folders that already have metrics.json. By default PNG samples are enough.",
    )
    return parser


def parse_run_dirs(runs_root, text):
    if text.strip():
        return [runs_root / item.strip() for item in text.split(",") if item.strip()]
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def has_saved_samples(path):
    return any(sample.is_dir() and (sample / "gt_00.png").exists() and (sample / "pd_00.png").exists()
               for sample in path.glob("sample_*"))


def discover_result_dirs(run_dirs, only_with_metrics):
    result_dirs = []
    for run_dir in run_dirs:
        results_dir = run_dir / "results"
        if not results_dir.exists():
            continue
        for result_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
            if only_with_metrics and not (result_dir / "metrics.json").exists():
                continue
            if has_saved_samples(result_dir):
                result_dirs.append((run_dir, result_dir))
    return result_dirs


def variant_suffix(run_name):
    lowered = run_name.lower()
    parts = []
    if "leadtime" in lowered or "lead_time" in lowered:
        parts.append("Lead-time")
    if "tendency" in lowered:
        parts.append("Tendency")
    if "attn" in lowered or "attention" in lowered:
        parts.append("Attention")
    return " ".join(parts)


def label_for(run_dir, result_dir, used_labels):
    base = RESULT_LABELS.get(result_dir.name, result_dir.name)
    suffix = variant_suffix(run_dir.name)
    if suffix and suffix.lower() not in base.lower():
        label = f"{base} {suffix}"
    else:
        label = base
    if run_dir.name == "north_china_3h" and result_dir.name == "radar_3h":
        label = "Radar-only"
    original = label
    index = 2
    while label in used_labels:
        label = f"{original} ({index})"
        index += 1
    used_labels.add(label)
    return label


def frame_index(path):
    stem = path.stem
    return int(stem.rsplit("_", 1)[1])


def sequence_paths(sample_dir, prefix):
    return sorted(sample_dir.glob(f"{prefix}_*.png"), key=frame_index)


def png_to_field(path, args):
    arr = np.array(Image.open(path).convert("L"), dtype="float32")
    span = max(args.pixel_max - args.pixel_min, 1e-6)
    if args.no_invert:
        field = (arr - args.pixel_min) / span * args.intensity_scale
    else:
        field = (args.pixel_max - arr) / span * args.intensity_scale
    return np.clip(field, 0.0, args.intensity_scale)


def load_sequence(sample_dir, prefix, args):
    paths = sequence_paths(sample_dir, prefix)
    if not paths:
        return None
    return np.stack([png_to_field(path, args) for path in paths], axis=0)


def load_saved_batches(result_dir, args):
    batches = []
    for sample_dir in sorted(result_dir.glob("sample_*")):
        target = load_sequence(sample_dir, "gt", args)
        pred = load_sequence(sample_dir, "pd", args)
        if target is None or pred is None:
            continue
        length = min(target.shape[0], pred.shape[0])
        target = target[:length]
        pred = pred[:length]
        persistence = load_sequence(sample_dir, "ps", args)
        if persistence is None:
            input_seq = load_sequence(sample_dir, "input", args)
            if input_seq is None or input_seq.shape[0] == 0:
                persistence = np.zeros_like(target)
            else:
                persistence = np.repeat(input_seq[-1:][:, ...], length, axis=0)
        persistence = persistence[:length]
        if persistence.shape[0] < length:
            persistence = np.pad(
                persistence,
                ((0, length - persistence.shape[0]), (0, 0), (0, 0)),
                mode="edge",
            )
        batches.append((sample_dir, pred, target, persistence))
    return batches


def compute_quantile_info(batches, quantiles, rain_min, intensity_scale):
    values = []
    total_count = 0
    dry_count = 0
    for _, _, target, _ in batches:
        total_count += int(target.size)
        rainy = target[target > rain_min]
        dry_count += int(target.size - rainy.size)
        if rainy.size:
            values.append(rainy.reshape(-1))
    if not values:
        return {
            "thresholds": {quantile_label(q): 0.0 for q in quantiles},
            "rain_min": rain_min,
            "rainy_count": 0,
            "dry_count": dry_count,
            "total_count": total_count,
            "source": "saved_target_pixels_above_rain_min",
        }
    rainy_values = np.concatenate(values)
    thresholds = {
        quantile_label(q): float(np.quantile(rainy_values, min(max(q, 0.0), 1.0)))
        for q in quantiles
    }
    return {
        "thresholds": thresholds,
        "rain_min": rain_min,
        "rainy_count": int(rainy_values.size),
        "dry_count": dry_count,
        "total_count": total_count,
        "source": "saved_target_pixels_above_rain_min",
    }


def recompute_metrics(result_dir, args):
    batches = load_saved_batches(result_dir, args)
    if not batches:
        raise ValueError(f"No saved gt_/pd_ sample PNGs found in {result_dir}")

    pred_length = min(item[1].shape[0] for item in batches)
    thresholds = parse_thresholds(args.metric_thresholds)
    neighborhood_thresholds = parse_thresholds(args.neighborhood_metric_thresholds or args.metric_thresholds)
    horizon_bins = parse_horizon_bins(args.horizon_bins)
    psd_lead_minutes = parse_float_list(args.psd_lead_minutes)
    psd_wavelengths = parse_float_list(args.psd_wavelengths)
    cra_thresholds = parse_thresholds(args.cra_thresholds)
    cra_lead_minutes = parse_float_list(args.cra_lead_minutes)
    object_thresholds = parse_thresholds(args.object_thresholds)
    extreme_quantiles = parse_float_list(args.extreme_quantiles)
    intensity_bin_quantiles = parse_float_list(args.intensity_bin_quantiles)
    quantile_info = compute_quantile_info(
        batches,
        sorted(set(extreme_quantiles + intensity_bin_quantiles)),
        args.extreme_rain_min,
        args.intensity_scale,
    )
    extreme_items = threshold_items_from_quantiles(
        quantile_info, [quantile_label(q) for q in extreme_quantiles]
    )
    intensity_bins = build_intensity_bins(quantile_info, args.extreme_rain_min)
    fss_items = threshold_items_from_quantiles(
        quantile_info, [quantile_label(q) for q in parse_float_list(args.fss_quantiles)]
    )
    fss_neighborhood_sizes = parse_int_list(args.fss_neighborhood_sizes)

    model_totals = init_scalar_totals()
    persistence_totals = init_scalar_totals()
    model_event_counts = init_event_counts(thresholds)
    persistence_event_counts = init_event_counts(thresholds)
    model_extreme_event_counts = init_labeled_event_counts(extreme_items)
    persistence_extreme_event_counts = init_labeled_event_counts(extreme_items)
    model_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    persistence_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    model_lead_totals = init_lead_totals(pred_length)
    persistence_lead_totals = init_lead_totals(pred_length)
    model_horizon_totals = init_horizon_totals(horizon_bins)
    persistence_horizon_totals = init_horizon_totals(horizon_bins)
    model_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    persistence_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    model_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    persistence_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    psd_totals = init_psd_totals(psd_lead_minutes, psd_wavelengths)
    model_pearson_totals = init_pearson_totals(pred_length)
    persistence_pearson_totals = init_pearson_totals(pred_length)
    model_eventwise = init_eventwise_store(thresholds)
    persistence_eventwise = init_eventwise_store(thresholds)
    model_cra = init_cra_store(cra_thresholds, cra_lead_minutes)
    persistence_cra = init_cra_store(cra_thresholds, cra_lead_minutes)
    model_objects = init_object_store(object_thresholds, pred_length)
    persistence_objects = init_object_store(object_thresholds, pred_length)

    device = torch.device(args.device)
    for _, pred_np, target_np, persistence_np in batches:
        pred = torch.from_numpy(pred_np[:pred_length]).unsqueeze(0).to(device)
        target = torch.from_numpy(target_np[:pred_length]).unsqueeze(0).to(device)
        persistence = torch.from_numpy(persistence_np[:pred_length]).unsqueeze(0).to(device)

        update_scalar_totals(model_totals, pred, target)
        update_scalar_totals(persistence_totals, persistence, target)
        update_lead_and_horizon(model_lead_totals, model_horizon_totals, pred, target, args.frame_minutes, horizon_bins)
        update_lead_and_horizon(
            persistence_lead_totals,
            persistence_horizon_totals,
            persistence,
            target,
            args.frame_minutes,
            horizon_bins,
        )
        update_event_counts(model_event_counts, pred, target, thresholds)
        update_event_counts(persistence_event_counts, persistence, target, thresholds)
        update_labeled_event_counts(model_extreme_event_counts, pred, target, extreme_items)
        update_labeled_event_counts(persistence_extreme_event_counts, persistence, target, extreme_items)
        update_neighborhood_event_counts(
            model_neighborhood_counts,
            pred,
            target,
            neighborhood_thresholds,
            args.neighborhood_size,
        )
        update_neighborhood_event_counts(
            persistence_neighborhood_counts,
            persistence,
            target,
            neighborhood_thresholds,
            args.neighborhood_size,
        )
        update_intensity_bin_totals(model_intensity_bin_totals, pred, target, intensity_bins)
        update_intensity_bin_totals(persistence_intensity_bin_totals, persistence, target, intensity_bins)
        update_fss_totals(model_fss_totals, pred, target, fss_items, fss_neighborhood_sizes)
        update_fss_totals(persistence_fss_totals, persistence, target, fss_items, fss_neighborhood_sizes)
        update_psd_totals(
            psd_totals,
            pred,
            target,
            persistence,
            args.frame_minutes,
            psd_lead_minutes,
            psd_wavelengths,
            args.grid_km,
        )
        update_pearson_totals(model_pearson_totals, pred, target)
        update_pearson_totals(persistence_pearson_totals, persistence, target)
        update_eventwise_store(model_eventwise, pred, target, thresholds)
        update_eventwise_store(persistence_eventwise, persistence, target, thresholds)
        update_cra_store(model_cra, pred, target, args.frame_minutes, cra_lead_minutes, cra_thresholds, args.cra_max_shift, args.grid_km)
        update_cra_store(persistence_cra, persistence, target, args.frame_minutes, cra_lead_minutes, cra_thresholds, args.cra_max_shift, args.grid_km)
        update_object_store(model_objects, pred, target, object_thresholds, args.object_min_area, args.object_iou_threshold, args.grid_km)
        update_object_store(persistence_objects, persistence, target, object_thresholds, args.object_min_area, args.object_iou_threshold, args.grid_km)

    model_neighborhood_metrics = finalize_neighborhood_metrics(model_neighborhood_counts)
    persistence_neighborhood_metrics = finalize_neighborhood_metrics(persistence_neighborhood_counts)
    metrics = {
        "model": finalize_scalar_totals(model_totals),
        "persistence": finalize_scalar_totals(persistence_totals),
        "samples": len(batches),
        "saved_samples": len(batches),
        "source": {
            "type": "saved_png_samples",
            "result_dir": str(result_dir),
            "note": "Metrics are recomputed only over saved sample_* PNG sequences.",
        },
        "units": {
            "prediction": "mm/h",
            "thresholds": "mm/h",
            "pixel_mapping": "255->0, 0->intensity_scale when invert is true",
            "intensity_scale": args.intensity_scale,
        },
        "thresholds": thresholds,
        "extreme_thresholds": quantile_info,
        "neighborhood_thresholds": neighborhood_thresholds,
        "neighborhood_size": args.neighborhood_size,
        "fss_neighborhood_sizes": fss_neighborhood_sizes,
        "cra_thresholds": cra_thresholds,
        "cra_lead_minutes": cra_lead_minutes,
        "cra_max_shift": args.cra_max_shift,
        "object_thresholds": object_thresholds,
        "object_min_area": args.object_min_area,
        "object_iou_threshold": args.object_iou_threshold,
        "frame_minutes": args.frame_minutes,
        "lead_time_metrics": {
            "model": finalize_lead_metrics(model_lead_totals, args.frame_minutes),
            "persistence": finalize_lead_metrics(persistence_lead_totals, args.frame_minutes),
        },
        "horizon_metrics": {
            "model": finalize_horizon_metrics(model_horizon_totals),
            "persistence": finalize_horizon_metrics(persistence_horizon_totals),
        },
        "event_metrics": {
            "model": finalize_event_metrics(model_event_counts),
            "persistence": finalize_event_metrics(persistence_event_counts),
        },
        "extreme_event_metrics": {
            "model": finalize_labeled_event_metrics(model_extreme_event_counts),
            "persistence": finalize_labeled_event_metrics(persistence_extreme_event_counts),
        },
        "intensity_bin_metrics": {
            "model": finalize_intensity_bin_metrics(model_intensity_bin_totals),
            "persistence": finalize_intensity_bin_metrics(persistence_intensity_bin_totals),
        },
        "neighborhood_event_metrics": {
            "model": model_neighborhood_metrics,
            "persistence": persistence_neighborhood_metrics,
        },
        "fss": {
            "model": finalize_fss_metrics(model_fss_totals, fss_items, args.grid_km),
            "persistence": finalize_fss_metrics(persistence_fss_totals, fss_items, args.grid_km),
        },
        "neighborhood_score": {
            "model": average_neighborhood_score(model_neighborhood_metrics),
            "persistence": average_neighborhood_score(persistence_neighborhood_metrics),
        },
        "psd": finalize_psd_metrics(psd_totals, psd_wavelengths, args.grid_km),
        "pearson": {
            "model": finalize_pearson_totals(model_pearson_totals, args.frame_minutes),
            "persistence": finalize_pearson_totals(persistence_pearson_totals, args.frame_minutes),
        },
        "eventwise": {
            "model": summarize_eventwise_store(model_eventwise),
            "persistence": summarize_eventwise_store(persistence_eventwise),
        },
        "cra": {
            "model": summarize_cra_store(model_cra),
            "persistence": summarize_cra_store(persistence_cra),
        },
        "object_metrics": {
            "model": summarize_object_store(model_objects, args.frame_minutes),
            "persistence": summarize_object_store(persistence_objects, args.frame_minutes),
        },
    }
    return metrics


def save_sample_grid(entries, out_dir):
    if not entries:
        return
    sample_entries = [(label, result_dir / "sample_0000") for label, result_dir, _ in entries]
    sample_entries = [(label, path) for label, path in sample_entries if path.exists()]
    if not sample_entries:
        return
    reference = sample_entries[0][1]
    preferred = [
        ("t+1\n0.1h", 0),
        ("t+10\n1.0h", 9),
        ("t+20\n2.0h", 19),
        ("t+25\n2.5h", 24),
        ("t+30\n3.0h", 29),
    ]
    columns = [(label, idx) for label, idx in preferred if (reference / f"gt_{idx:02d}.png").exists()]
    if not columns:
        return

    rows = [
        ("Ground truth", lambda i: open_rgb(reference / f"gt_{i:02d}.png")),
        ("Persistence", lambda i: open_rgb(reference / f"ps_{i:02d}.png")),
    ]
    for label, sample_dir in sample_entries:
        rows.append((label, lambda i, p=sample_dir: open_rgb(p / f"pd_{i:02d}.png")))
        if (sample_dir / "c_00.png").exists():
            rows.append((f"Coupling {label}", lambda i, p=sample_dir: colorize_gray(p / f"c_{i:02d}.png", cmap="magma")))
        if (sample_dir / "s_00.png").exists():
            rows.append((f"Support {label}", lambda i, p=sample_dir: colorize_gray(p / f"s_{i:02d}.png", cmap="magma")))
        if (sample_dir / "a_00.png").exists():
            rows.append((f"Attention {label}", lambda i, p=sample_dir: colorize_gray(p / f"a_{min(i, 8):02d}.png", cmap="viridis", stretch=True)))
        if (sample_dir / "oc_00.png").exists():
            rows.append((f"Object center {label}", lambda i, p=sample_dir: colorize_gray(p / f"oc_{i:02d}.png", cmap="magma", stretch=True)))
        if (sample_dir / "om_00.png").exists():
            rows.append((f"Object mask {label}", lambda i, p=sample_dir: colorize_gray(p / f"om_{i:02d}.png", cmap="magma", stretch=True)))

    cell_w, cell_h = 96, 96
    label_w, top_h = 188, 42
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
    canvas.save(out_dir / "sample_0000_recomputed_grid.png")


def save_cumulative_rainfall(entries, out_dir, args):
    sample_entries = [(label, result_dir / "sample_0000") for label, result_dir, _ in entries]
    sample_entries = [(label, path) for label, path in sample_entries if path.exists()]
    if not sample_entries:
        return
    reference = sample_entries[0][1]
    target = load_sequence(reference, "gt", args)
    persistence = load_sequence(reference, "ps", args)
    if target is None:
        return
    x = np.arange(1, target.shape[0] + 1) * args.frame_minutes / 60.0
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=180)
    ax.plot(x, np.cumsum(target.mean(axis=(1, 2))), color="0.15", linewidth=2.6, label="Ground truth")
    if persistence is not None:
        length = min(len(x), persistence.shape[0])
        ax.plot(x[:length], np.cumsum(persistence[:length].mean(axis=(1, 2))), color="0.45", linestyle="--", linewidth=2.0, label="Persistence")
    for label, sample_dir in sample_entries:
        pred = load_sequence(sample_dir, "pd", args)
        if pred is None:
            continue
        length = min(len(x), pred.shape[0])
        ax.plot(x[:length], np.cumsum(pred[:length].mean(axis=(1, 2))), linewidth=2.0, label=label)
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("Cumulative domain-mean rain rate")
    ax.set_title("Sample cumulative rainfall curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_0000_cumulative_rainfall.png")
    plt.close(fig)


def render_error_image(error, vmax):
    cm = plt.get_cmap("inferno")
    arr = np.clip(error / max(vmax, 1e-6), 0.0, 1.0)
    return Image.fromarray((cm(arr)[..., :3] * 255).astype("uint8"))


def save_spatial_error_maps(entries, out_dir, args):
    sample_entries = [(label, result_dir / "sample_0000") for label, result_dir, _ in entries]
    sample_entries = [(label, path) for label, path in sample_entries if path.exists()]
    if not sample_entries:
        return
    reference = sample_entries[0][1]
    target = load_sequence(reference, "gt", args)
    if target is None:
        return
    preferred = [
        ("t+1\n0.1h", 0),
        ("t+10\n1.0h", 9),
        ("t+20\n2.0h", 19),
        ("t+30\n3.0h", 29),
    ]
    columns = [(label, idx) for label, idx in preferred if idx < target.shape[0]]
    if not columns:
        return
    rows = []
    for label, sample_dir in sample_entries:
        pred = load_sequence(sample_dir, "pd", args)
        if pred is None:
            continue
        rows.append((label, pred))
    if not rows:
        return
    vmax = max(
        float(np.nanmax(np.abs(pred[:target.shape[0]] - target[:pred.shape[0]])))
        for _, pred in rows
    )
    cell_w, cell_h = 96, 96
    label_w, top_h = 188, 42
    canvas = Image.new("RGB", (label_w + len(columns) * cell_w, top_h + len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (label, _) in enumerate(columns):
        draw.multiline_text((label_w + c * cell_w + 27, 6), label, fill="black", align="center", spacing=2)
    for r, (label, pred) in enumerate(rows):
        y = top_h + r * cell_h
        draw.text((8, y + 40), label, fill="black")
        for c, (_, idx) in enumerate(columns):
            if idx >= pred.shape[0]:
                continue
            error = np.abs(pred[idx] - target[idx])
            canvas.paste(render_error_image(error, vmax), (label_w + c * cell_w, y))
    canvas.save(out_dir / "sample_0000_abs_error_maps.png")


def save_summary(metrics_by_label, manifest_rows, out_dir):
    summary = OrderedDict()
    for label, metrics in metrics_by_label.items():
        summary[label] = {
            "model": metrics.get("model"),
            "persistence": metrics.get("persistence"),
            "samples": metrics.get("samples"),
            "horizon_metrics": metrics.get("horizon_metrics", {}).get("model"),
            "event_metrics": metrics.get("event_metrics", {}).get("model"),
            "neighborhood_event_metrics": metrics.get("neighborhood_event_metrics", {}).get("model"),
            "neighborhood_score": metrics.get("neighborhood_score", {}).get("model"),
            "pearson": metrics.get("pearson", {}).get("model"),
            "eventwise": metrics.get("eventwise", {}).get("model"),
            "cra": metrics.get("cra", {}).get("model"),
            "object_metrics": metrics.get("object_metrics", {}).get("model"),
        }
    with open(out_dir / "summary_recomputed.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)


def write_manifest_csv(rows, path):
    fieldnames = [
        "label",
        "run_dir",
        "result_dir",
        "metrics_path",
        "samples",
        "mae",
        "rmse",
        "persistence_mae",
        "persistence_rmse",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else runs_root / "recomputed_reports" / "comparison_3h"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = parse_run_dirs(runs_root, args.run_dirs)
    result_dirs = discover_result_dirs(run_dirs, args.only_with_metrics)
    if not result_dirs:
        raise SystemExit(f"No saved result folders with sample_*/gt_00.png and pd_00.png found under {runs_root}")

    used_labels = set()
    metrics_by_label = OrderedDict()
    entries = []
    manifest_rows = []
    for run_dir, result_dir in result_dirs:
        label = label_for(run_dir, result_dir, used_labels)
        print(f"recomputing {label}: {result_dir}", flush=True)
        metrics = recompute_metrics(result_dir, args)
        metrics_path = result_dir / args.metrics_name
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        metrics_by_label[label] = metrics
        entries.append((label, result_dir, metrics))
        manifest_rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "result_dir": str(result_dir),
                "metrics_path": str(metrics_path),
                "samples": metrics["samples"],
                "mae": metrics["model"]["mae"],
                "rmse": metrics["model"]["rmse"],
                "persistence_mae": metrics["persistence"]["mae"],
                "persistence_rmse": metrics["persistence"]["rmse"],
            }
        )

    save_lead_curves(metrics_by_label, output_dir)
    save_horizon_bars(metrics_by_label, output_dir)
    save_threshold_metrics(metrics_by_label, output_dir)
    save_extreme_threshold_metrics(metrics_by_label, output_dir)
    save_neighborhood_csi(metrics_by_label, output_dir)
    save_fss(metrics_by_label, output_dir)
    save_pearson(metrics_by_label, output_dir)
    save_cra(metrics_by_label, output_dir)
    save_object_metrics(metrics_by_label, output_dir)
    save_intensity_bin_metrics(metrics_by_label, output_dir)
    save_intensity_bin_improvement(metrics_by_label, output_dir)
    save_psd_plots(metrics_by_label, output_dir)
    save_psd_error(metrics_by_label, output_dir)
    save_sample_grid(entries, output_dir)
    save_cumulative_rainfall(entries, output_dir, args)
    save_spatial_error_maps(entries, output_dir, args)
    save_summary(metrics_by_label, manifest_rows, output_dir)
    write_manifest_csv(manifest_rows, output_dir / args.manifest_name)
    print(f"saved recomputed comparison report to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
