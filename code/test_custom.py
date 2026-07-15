import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from nowcasting.experiments.common import (
    add_model_runtime_args,
    build_generator,
    load_model_state,
    make_png_dataloader,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Test a custom NowcastNet checkpoint")
    parser.add_argument("--data_root", type=str, default="../data/DATA_2025_S/RADAR_2025_S")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../results/custom_test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="NowcastNet")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--lead_time_embed_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--num_save_samples", type=int, default=10)
    parser.add_argument("--intensity_scale", type=float, default=128.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--metric_thresholds", type=str, default="1,5,10,20,40")
    parser.add_argument("--neighborhood_metric_thresholds", type=str, default="")
    parser.add_argument("--neighborhood_size", type=int, default=5)
    parser.add_argument("--extreme_quantiles", type=str, default="0.9,0.95,0.99")
    parser.add_argument("--extreme_rain_min", type=float, default=0.1)
    parser.add_argument("--quantile_bins", type=int, default=4096)
    parser.add_argument("--intensity_bin_quantiles", type=str, default="0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--fss_quantiles", type=str, default="0.95,0.99")
    parser.add_argument("--fss_neighborhood_sizes", type=str, default="1,3,5,9,15")
    parser.add_argument("--num_extreme_cases", type=int, default=5)
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--horizon_bins", type=str, default="0-1,1-2,2-3,3-6")
    parser.add_argument("--psd_lead_minutes", type=str, default="60,120,180")
    parser.add_argument("--psd_wavelengths", type=str, default="4,8,16,32,64")
    parser.add_argument("--grid_km", type=float, default=1.0)
    parser.add_argument("--cra_thresholds", type=str, default="16")
    parser.add_argument("--cra_lead_minutes", type=str, default="60,120,180")
    parser.add_argument("--cra_max_shift", type=int, default=12)
    parser.add_argument("--no_invert", action="store_true")
    return parser


load_state = load_model_state


def parse_thresholds(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_horizon_bins(text):
    bins = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        start, end = item.split("-")
        bins.append((float(start), float(end), "{}h-{}h".format(start, end)))
    return bins


def parse_float_list(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text):
    return [int(item) for item in text.split(",") if item.strip()]


def quantile_label(value):
    return "P{:g}".format(float(value) * 100.0)


def threshold_key(value):
    return str(float(value))


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def to_png(field, intensity_scale, pixel_min=0.0, pixel_max=255.0, invert=True):
    arr = np.clip(field, 0.0, intensity_scale) / max(intensity_scale, 1e-6)
    span = max(pixel_max - pixel_min, 1e-6)
    if invert:
        arr = pixel_max - arr * span
    else:
        arr = pixel_min + arr * span
    return np.clip(arr, 0, 255).astype("uint8")


def save_sequence(folder, prefix, seq, intensity_scale, pixel_min, pixel_max, invert):
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(seq):
        path = folder / "{}{:02d}.png".format(prefix, i)
        image = to_png(frame, intensity_scale, pixel_min, pixel_max, invert)
        if cv2 is not None:
            cv2.imwrite(str(path), image)
        else:
            Image.fromarray(image).save(path)


def save_color_sequence(folder, prefix, seq):
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(seq):
        path = folder / "{}{:02d}.png".format(prefix, i)
        Image.fromarray(frame.astype("uint8"), mode="RGB").save(path)


def event_map_sequence(pred, target, threshold):
    pred_event = pred >= threshold
    target_event = target >= threshold
    maps = np.zeros(pred_event.shape + (3,), dtype="uint8")
    hit = np.logical_and(pred_event, target_event)
    miss = np.logical_and(~pred_event, target_event)
    false_alarm = np.logical_and(pred_event, ~target_event)
    maps[hit] = np.array([50, 180, 80], dtype="uint8")
    maps[miss] = np.array([45, 105, 210], dtype="uint8")
    maps[false_alarm] = np.array([220, 70, 60], dtype="uint8")
    return maps


def init_event_counts(thresholds):
    return {
        str(thr): {"hit": 0, "miss": 0, "false_alarm": 0, "correct_negative": 0}
        for thr in thresholds
    }


def update_event_counts(counts, pred, target, thresholds):
    for thr in thresholds:
        key = str(thr)
        pred_event = pred >= thr
        target_event = target >= thr
        counts[key]["hit"] += torch.logical_and(pred_event, target_event).sum().item()
        counts[key]["miss"] += torch.logical_and(~pred_event, target_event).sum().item()
        counts[key]["false_alarm"] += torch.logical_and(pred_event, ~target_event).sum().item()
        counts[key]["correct_negative"] += torch.logical_and(~pred_event, ~target_event).sum().item()


def finalize_event_metrics(counts):
    metrics = {}
    for threshold, values in counts.items():
        hit = values["hit"]
        miss = values["miss"]
        false_alarm = values["false_alarm"]
        correct_negative = values["correct_negative"]
        csi_den = hit + miss + false_alarm
        pod_den = hit + miss
        far_den = hit + false_alarm
        bias_den = hit + miss
        total = hit + miss + false_alarm + correct_negative
        random_hit = (hit + miss) * (hit + false_alarm) / total if total else 0.0
        ets_den = hit + miss + false_alarm - random_hit
        hss_den = (hit + miss) * (miss + correct_negative) + (hit + false_alarm) * (false_alarm + correct_negative)
        f1_den = 2 * hit + miss + false_alarm
        metrics[threshold] = {
            "threshold": float(threshold),
            "hit": hit,
            "miss": miss,
            "false_alarm": false_alarm,
            "correct_negative": correct_negative,
            "csi": safe_divide(hit, csi_den),
            "pod": safe_divide(hit, pod_den),
            "far": safe_divide(false_alarm, far_den),
            "f1": safe_divide(2 * hit, f1_den),
            "bias": safe_divide(hit + false_alarm, bias_den),
            "ets": safe_divide(hit - random_hit, ets_den),
            "hss": safe_divide(2 * (hit * correct_negative - false_alarm * miss), hss_den),
        }
    return metrics


def init_labeled_event_counts(threshold_items):
    return {
        label: {
            "threshold": float(threshold),
            "hit": 0,
            "miss": 0,
            "false_alarm": 0,
            "correct_negative": 0,
        }
        for label, threshold in threshold_items
    }


def update_labeled_event_counts(counts, pred, target, threshold_items):
    for label, threshold in threshold_items:
        pred_event = pred >= threshold
        target_event = target >= threshold
        counts[label]["hit"] += torch.logical_and(pred_event, target_event).sum().item()
        counts[label]["miss"] += torch.logical_and(~pred_event, target_event).sum().item()
        counts[label]["false_alarm"] += torch.logical_and(pred_event, ~target_event).sum().item()
        counts[label]["correct_negative"] += torch.logical_and(~pred_event, ~target_event).sum().item()


def finalize_labeled_event_metrics(counts):
    threshold_counts = {
        threshold_key(values["threshold"]): {
            "hit": values["hit"],
            "miss": values["miss"],
            "false_alarm": values["false_alarm"],
            "correct_negative": values["correct_negative"],
        }
        for values in counts.values()
    }
    by_threshold = finalize_event_metrics(threshold_counts)
    metrics = {}
    for label, values in counts.items():
        item = by_threshold[threshold_key(values["threshold"])]
        item["label"] = label
        metrics[label] = item
    return metrics


def update_neighborhood_event_counts(counts, pred, target, thresholds, neighborhood_size):
    padding = neighborhood_size // 2
    for thr in thresholds:
        key = str(thr)
        pred_event = pred >= thr
        target_event = target >= thr
        pred_neighborhood = torch.nn.functional.max_pool2d(
            pred_event.reshape(-1, 1, pred_event.shape[-2], pred_event.shape[-1]).float(),
            kernel_size=neighborhood_size,
            stride=1,
            padding=padding,
        ).reshape_as(pred_event).bool()
        target_neighborhood = torch.nn.functional.max_pool2d(
            target_event.reshape(-1, 1, target_event.shape[-2], target_event.shape[-1]).float(),
            kernel_size=neighborhood_size,
            stride=1,
            padding=padding,
        ).reshape_as(target_event).bool()
        counts[key]["hit"] += torch.logical_and(pred_event, target_neighborhood).sum().item()
        counts[key]["miss"] += torch.logical_and(target_event, ~pred_neighborhood).sum().item()
        counts[key]["false_alarm"] += torch.logical_and(pred_event, ~target_neighborhood).sum().item()


def finalize_neighborhood_metrics(counts):
    metrics = {}
    for threshold, values in counts.items():
        hit = values["hit"]
        miss = values["miss"]
        false_alarm = values["false_alarm"]
        den = hit + miss + false_alarm
        metrics[threshold] = {
            "hit": hit,
            "miss": miss,
            "false_alarm": false_alarm,
            "csin": hit / den if den else 0.0,
        }
    return metrics


def average_neighborhood_score(metrics):
    values = [item["csin"] for item in metrics.values()]
    return float(sum(values) / len(values)) if values else 0.0


def init_scalar_totals():
    return {"abs": 0.0, "sq": 0.0, "err": 0.0, "pred_sum": 0.0, "target_sum": 0.0, "count": 0}


def update_scalar_totals(totals, pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        if pred.numel() == 0:
            return
    diff = pred - target
    totals["abs"] += diff.abs().sum().item()
    totals["sq"] += (diff ** 2).sum().item()
    totals["err"] += diff.sum().item()
    totals["pred_sum"] += pred.sum().item()
    totals["target_sum"] += target.sum().item()
    totals["count"] += diff.numel()


def finalize_scalar_totals(totals):
    count = max(totals["count"], 1)
    mse = totals["sq"] / count
    rmse = mse ** 0.5
    target_mean = totals["target_sum"] / count
    return {
        "mae": totals["abs"] / count,
        "mse": mse,
        "rmse": rmse,
        "bias": totals["err"] / count,
        "mean_error": totals["err"] / count,
        "bias_ratio": safe_divide(totals["pred_sum"], totals["target_sum"]),
        "normalized_error": safe_divide(rmse, target_mean),
        "relative_bias": totals["err"] / totals["target_sum"] if abs(totals["target_sum"]) > 1e-12 else 0.0,
        "count": totals["count"],
    }


def nan_summary(values):
    arr = np.array([value for value in values if value is not None], dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def scalar_metrics_for_arrays(pred, target):
    pred = pred.detach().float()
    target = target.detach().float()
    diff = pred - target
    mse = float((diff ** 2).mean().item())
    rmse = mse ** 0.5
    target_mean = float(target.mean().item())
    return {
        "mae": float(diff.abs().mean().item()),
        "mse": mse,
        "rmse": rmse,
        "bias": float(diff.mean().item()),
        "mean_error": float(diff.mean().item()),
        "bias_ratio": safe_divide(float(pred.mean().item()), target_mean),
        "normalized_error": safe_divide(rmse, target_mean),
        "count": int(diff.numel()),
    }


def pearson_corr(pred, target):
    pred = pred.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    pred = pred - pred.mean()
    target = target - target.mean()
    den = torch.sqrt((pred ** 2).sum()) * torch.sqrt((target ** 2).sum())
    if float(den.item()) <= 1e-12:
        return float("nan")
    return float(((pred * target).sum() / den).item())


def init_pearson_totals(pred_length):
    return {
        "overall": {"sum": 0.0, "count": 0},
        "lead": [{"sum": 0.0, "count": 0} for _ in range(pred_length)],
    }


def update_pearson_totals(totals, pred, target):
    for batch_index in range(pred.shape[0]):
        value = pearson_corr(pred[batch_index], target[batch_index])
        if np.isfinite(value):
            totals["overall"]["sum"] += value
            totals["overall"]["count"] += 1
        for lead in range(pred.shape[1]):
            lead_value = pearson_corr(pred[batch_index, lead], target[batch_index, lead])
            if np.isfinite(lead_value):
                totals["lead"][lead]["sum"] += lead_value
                totals["lead"][lead]["count"] += 1


def finalize_pearson_totals(totals, frame_minutes):
    overall_count = totals["overall"]["count"]
    return {
        "overall": safe_divide(totals["overall"]["sum"], overall_count),
        "count": overall_count,
        "lead_time": [
            {
                "lead_index": index + 1,
                "lead_minutes": (index + 1) * frame_minutes,
                "pearson": safe_divide(item["sum"], item["count"]),
                "count": item["count"],
            }
            for index, item in enumerate(totals["lead"])
        ],
    }


def event_metrics_for_arrays(pred, target, thresholds):
    counts = init_event_counts(thresholds)
    update_event_counts(counts, pred.unsqueeze(0), target.unsqueeze(0), thresholds)
    return finalize_event_metrics(counts)


def init_eventwise_store(thresholds):
    return {
        "scalar": [],
        "pearson": [],
        "thresholds": {threshold_key(thr): [] for thr in thresholds},
    }


def update_eventwise_store(store, pred, target, thresholds):
    for batch_index in range(pred.shape[0]):
        sample_pred = pred[batch_index]
        sample_target = target[batch_index]
        sample_scalar = scalar_metrics_for_arrays(sample_pred, sample_target)
        sample_scalar["pearson"] = pearson_corr(sample_pred, sample_target)
        store["scalar"].append(sample_scalar)
        store["pearson"].append(sample_scalar["pearson"])
        sample_events = event_metrics_for_arrays(sample_pred, sample_target, thresholds)
        for key, values in sample_events.items():
            store["thresholds"][key].append(values)


def summarize_eventwise_store(store):
    scalar_keys = ("mae", "rmse", "bias", "bias_ratio", "normalized_error", "pearson")
    threshold_keys = ("csi", "pod", "far", "f1", "bias", "ets", "hss")
    return {
        "scalar": {
            key: nan_summary([item.get(key) for item in store["scalar"]])
            for key in scalar_keys
        },
        "thresholds": {
            threshold: {
                key: nan_summary([item.get(key) for item in values])
                for key in threshold_keys
            }
            for threshold, values in store["thresholds"].items()
        },
    }


def init_lead_totals(pred_length):
    return [init_scalar_totals() for _ in range(pred_length)]


def init_horizon_totals(horizon_bins):
    return {label: init_scalar_totals() for _, _, label in horizon_bins}


def update_lead_and_horizon(lead_totals, horizon_totals, pred, target, frame_minutes, horizon_bins):
    for i in range(pred.shape[1]):
        update_scalar_totals(lead_totals[i], pred[:, i], target[:, i])
        lead_hours = (i + 1) * frame_minutes / 60.0
        for start, end, label in horizon_bins:
            if start < lead_hours <= end:
                update_scalar_totals(horizon_totals[label], pred[:, i], target[:, i])


def finalize_lead_metrics(lead_totals, frame_minutes):
    metrics = []
    for i, totals in enumerate(lead_totals):
        item = finalize_scalar_totals(totals)
        item["lead_index"] = i + 1
        item["lead_minutes"] = (i + 1) * frame_minutes
        metrics.append(item)
    return metrics


def finalize_horizon_metrics(horizon_totals):
    return {label: finalize_scalar_totals(totals) for label, totals in horizon_totals.items()}


def compute_target_quantile_thresholds(loader, quantiles, rain_min, intensity_scale, bins):
    if not quantiles:
        return {}
    bins = max(int(bins), 32)
    max_value = max(float(intensity_scale), float(rain_min) + 1e-6)
    hist = torch.zeros(bins, dtype=torch.float64)
    rainy_count = 0
    total_count = 0
    dry_count = 0
    for batch in loader:
        target = batch["target_frames"].float()
        total_count += target.numel()
        rainy = target[target > rain_min]
        dry_count += target.numel() - rainy.numel()
        if rainy.numel() == 0:
            continue
        rainy = rainy.clamp(0.0, max_value)
        hist += torch.histc(rainy, bins=bins, min=0.0, max=max_value).double()
        rainy_count += rainy.numel()
    if rainy_count == 0:
        return {
            "thresholds": {quantile_label(q): 0.0 for q in quantiles},
            "rain_min": rain_min,
            "rainy_count": 0,
            "dry_count": dry_count,
            "total_count": total_count,
            "source": "target_pixels_above_rain_min",
        }
    cdf = torch.cumsum(hist, dim=0)
    thresholds = {}
    for q in quantiles:
        rank = max(float(q), 0.0) * max(rainy_count - 1, 0) + 1
        index = int(torch.searchsorted(cdf, torch.tensor(rank, dtype=torch.float64)).item())
        index = min(max(index, 0), bins - 1)
        thresholds[quantile_label(q)] = (index + 0.5) / bins * max_value
    return {
        "thresholds": thresholds,
        "rain_min": rain_min,
        "rainy_count": rainy_count,
        "dry_count": dry_count,
        "total_count": total_count,
        "source": "target_pixels_above_rain_min",
        "histogram_bins": bins,
    }


def threshold_items_from_quantiles(quantile_info, selected_labels=None):
    thresholds = quantile_info.get("thresholds", {})
    if selected_labels:
        labels = [label for label in selected_labels if label in thresholds]
    else:
        labels = list(thresholds.keys())
    return [(label, float(thresholds[label])) for label in labels]


def build_intensity_bins(quantile_info, rain_min):
    thresholds = quantile_info.get("thresholds", {})
    required = ["P50", "P75", "P90", "P95", "P99"]
    if not all(label in thresholds for label in required):
        return []
    bins = [("dry", None, float(rain_min))]
    previous_label = "rain_min"
    previous_value = float(rain_min)
    for label in required:
        value = float(thresholds[label])
        bins.append(("{}-{}".format(previous_label, label), previous_value, value))
        previous_label = label
        previous_value = value
    bins.append(("gt-P99", float(thresholds["P99"]), None))
    return bins


def init_intensity_bin_totals(bins):
    return {label: init_scalar_totals() for label, _, _ in bins}


def update_intensity_bin_totals(totals, pred, target, bins):
    for label, low, high in bins:
        if low is None:
            mask = target <= high
        elif high is None:
            mask = target > low
        else:
            mask = torch.logical_and(target > low, target <= high)
        update_scalar_totals(totals[label], pred, target, mask)


def finalize_intensity_bin_metrics(totals):
    return {label: finalize_scalar_totals(values) for label, values in totals.items()}


def init_fss_totals(threshold_items, neighborhood_sizes):
    return {
        label: {
            str(size): {"num": 0.0, "den": 0.0, "count": 0}
            for size in neighborhood_sizes
        }
        for label, _ in threshold_items
    }


def update_fss_totals(totals, pred, target, threshold_items, neighborhood_sizes):
    for label, threshold in threshold_items:
        pred_event = (pred >= threshold).float()
        target_event = (target >= threshold).float()
        flat_pred = pred_event.reshape(-1, 1, pred_event.shape[-2], pred_event.shape[-1])
        flat_target = target_event.reshape(-1, 1, target_event.shape[-2], target_event.shape[-1])
        for size in neighborhood_sizes:
            padding = size // 2
            if size == 1:
                pred_fraction = flat_pred
                target_fraction = flat_target
            else:
                pred_fraction = torch.nn.functional.avg_pool2d(
                    flat_pred, kernel_size=size, stride=1, padding=padding, count_include_pad=False
                )
                target_fraction = torch.nn.functional.avg_pool2d(
                    flat_target, kernel_size=size, stride=1, padding=padding, count_include_pad=False
                )
            num = ((pred_fraction - target_fraction) ** 2).sum().item()
            den = (pred_fraction ** 2 + target_fraction ** 2).sum().item()
            key = str(size)
            totals[label][key]["num"] += num
            totals[label][key]["den"] += den
            totals[label][key]["count"] += pred_fraction.numel()


def finalize_fss_metrics(totals, threshold_items, grid_km):
    thresholds = {label: threshold for label, threshold in threshold_items}
    metrics = {"grid_km": grid_km, "thresholds": {}}
    for label, by_size in totals.items():
        metrics["thresholds"][label] = {
            "threshold": float(thresholds[label]),
            "neighborhoods": {},
        }
        for size, values in by_size.items():
            den = values["den"]
            metrics["thresholds"][label]["neighborhoods"][size] = {
                "fss": 1.0 - values["num"] / den if den else 0.0,
                "num": values["num"],
                "den": den,
                "count": values["count"],
                "size_pixels": int(size),
                "size_km": int(size) * float(grid_km),
            }
    return metrics


def init_extreme_cases(limit):
    return [] if limit > 0 else None


def update_extreme_cases(cases, limit, batch, pred, target, persistence, arrays, extreme_threshold):
    if cases is None or limit <= 0:
        return
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    persistence_np = persistence.detach().cpu().numpy()
    for i in range(target_np.shape[0]):
        extreme_pixels = int((target_np[i] >= extreme_threshold).sum()) if extreme_threshold > 0 else 0
        target_max = float(target_np[i].max())
        score = float(extreme_pixels) * 1000.0 + target_max
        item = {
            "score": score,
            "target_max": target_max,
            "extreme_pixels": extreme_pixels,
            "case_name": str(batch.get("case_name", [""] * target_np.shape[0])[i]),
            "start_file": str(batch.get("start_file", [""] * target_np.shape[0])[i]),
            "pred": pred_np[i].copy(),
            "target": target_np[i].copy(),
            "persistence": persistence_np[i].copy(),
        }
        for key, value in arrays.items():
            item[key] = value[i].copy()
        cases.append(item)
    cases.sort(key=lambda item: item["score"], reverse=True)
    del cases[limit:]


def save_extreme_cases(output_dir, cases, thresholds, args, invert):
    if not cases:
        return
    case_root = output_dir / "extreme_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    event_thresholds = {}
    for label in ("P95", "P99"):
        if label in thresholds:
            event_thresholds[label] = float(thresholds[label])
    for rank, item in enumerate(cases):
        folder = case_root / "case_{:04d}".format(rank)
        folder.mkdir(parents=True, exist_ok=True)
        save_sequence(folder, "gt_", item["target"], args.intensity_scale, args.pixel_min, args.pixel_max, invert)
        save_sequence(folder, "pd_", item["pred"], args.intensity_scale, args.pixel_min, args.pixel_max, invert)
        save_sequence(folder, "ps_", item["persistence"], args.intensity_scale, args.pixel_min, args.pixel_max, invert)
        if "input" in item:
            save_sequence(folder, "input_", item["input"], args.intensity_scale, args.pixel_min, args.pixel_max, invert)
        if "pwv" in item:
            save_sequence(folder, "pwv_", item["pwv"], args.pwv_intensity_scale, args.pwv_pixel_min, args.pwv_pixel_max, args.pwv_invert)
        for key, prefix in (("coupling", "c_"), ("support", "s_"), ("attention", "a_")):
            if key in item:
                save_sequence(folder, prefix, item[key], 1.0, 0.0, 255.0, False)
        abs_error = np.abs(item["pred"] - item["target"])
        save_sequence(folder, "err_", abs_error, args.intensity_scale, args.pixel_min, args.pixel_max, False)
        for label, threshold in event_thresholds.items():
            save_color_sequence(folder, "hmf_{}_".format(label.lower()), event_map_sequence(item["pred"], item["target"], threshold))
        metadata = {
            "rank": rank,
            "score": item["score"],
            "case_name": item["case_name"],
            "start_file": item["start_file"],
            "target_max": item["target_max"],
            "extreme_pixels": item["extreme_pixels"],
            "event_thresholds": event_thresholds,
        }
        with open(folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def init_psd_totals(lead_minutes, wavelengths):
    return {
        str(int(lead)): {
            "model": np.zeros(len(wavelengths), dtype="float64"),
            "target": np.zeros(len(wavelengths), dtype="float64"),
            "persistence": np.zeros(len(wavelengths), dtype="float64"),
            "model_log_mse": np.zeros(len(wavelengths), dtype="float64"),
            "persistence_log_mse": np.zeros(len(wavelengths), dtype="float64"),
            "count": 0,
        }
        for lead in lead_minutes
    }


def radial_psd(field, wavelengths, grid_km):
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    height, width = field.shape[-2:]
    fft = torch.fft.fft2(field)
    power = (fft.real * fft.real + fft.imag * fft.imag) / float(height * width)
    fy = torch.fft.fftfreq(height, d=grid_km, device=field.device)
    fx = torch.fft.fftfreq(width, d=grid_km, device=field.device)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(kx * kx + ky * ky)
    spectra = []
    for wavelength in wavelengths:
        center = 1.0 / max(float(wavelength), 1e-6)
        low = center / 2.0 ** 0.5
        high = center * 2.0 ** 0.5
        mask = (radius >= low) & (radius < high)
        if not bool(mask.any()):
            nearest = torch.argmin(torch.abs(radius - center))
            mask = torch.zeros_like(radius, dtype=torch.bool).flatten()
            mask[nearest] = True
            mask = mask.reshape_as(radius)
        spectra.append(power[:, mask].mean(dim=1))
    return torch.stack(spectra, dim=1)


def update_psd_totals(psd_totals, pred, target, persistence, frame_minutes, lead_minutes, wavelengths, grid_km):
    eps = 1e-8
    for lead in lead_minutes:
        index = int(round(float(lead) / frame_minutes)) - 1
        if index < 0 or index >= pred.shape[1]:
            continue
        model_spec = radial_psd(pred[:, index], wavelengths, grid_km)
        target_spec = radial_psd(target[:, index], wavelengths, grid_km)
        persistence_spec = radial_psd(persistence[:, index], wavelengths, grid_km)
        key = str(int(lead))
        psd_totals[key]["model"] += model_spec.sum(dim=0).detach().cpu().numpy()
        psd_totals[key]["target"] += target_spec.sum(dim=0).detach().cpu().numpy()
        psd_totals[key]["persistence"] += persistence_spec.sum(dim=0).detach().cpu().numpy()
        psd_totals[key]["model_log_mse"] += (
            (torch.log(model_spec + eps) - torch.log(target_spec + eps)) ** 2
        ).sum(dim=0).detach().cpu().numpy()
        psd_totals[key]["persistence_log_mse"] += (
            (torch.log(persistence_spec + eps) - torch.log(target_spec + eps)) ** 2
        ).sum(dim=0).detach().cpu().numpy()
        psd_totals[key]["count"] += pred.shape[0]


def finalize_psd_metrics(psd_totals, wavelengths, grid_km):
    metrics = {
        "grid_km": grid_km,
        "wavelengths": wavelengths,
        "lead_minutes": {},
    }
    for lead, totals in psd_totals.items():
        count = max(totals["count"], 1)
        metrics["lead_minutes"][lead] = {
            "target": (totals["target"] / count).tolist(),
            "model": (totals["model"] / count).tolist(),
            "persistence": (totals["persistence"] / count).tolist(),
            "model_log_rmse": np.sqrt(totals["model_log_mse"] / count).tolist(),
            "persistence_log_rmse": np.sqrt(totals["persistence_log_mse"] / count).tolist(),
        }
    return metrics


def shift_2d_zero(field, dx, dy):
    shifted = np.zeros_like(field)
    height, width = field.shape
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return shifted
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = field[src_y0:src_y1, src_x0:src_x1]
    return shifted


def largest_component_mask(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.sum() == 0:
        return mask.astype(bool)
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask.astype(bool)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    visited = np.zeros(mask.shape, dtype=bool)
    best_pixels = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(pixels) > len(best_pixels):
                best_pixels = pixels
    result = np.zeros(mask.shape, dtype=bool)
    for y, x in best_pixels:
        result[y, x] = True
    return result


def pearson_corr_np(a, b):
    a = np.asarray(a, dtype="float64").reshape(-1)
    b = np.asarray(b, dtype="float64").reshape(-1)
    if a.size == 0:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if den <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / den)


def cra_one_field(pred, target, threshold, max_shift, grid_km):
    pred_np = np.asarray(pred, dtype="float64")
    target_np = np.asarray(target, dtype="float64")
    obs_mask = largest_component_mask(target_np > threshold)
    pred_mask = largest_component_mask(pred_np > threshold)
    obs_pixels = int(obs_mask.sum())
    pred_pixels = int(pred_mask.sum())
    if obs_pixels == 0 or pred_pixels == 0:
        return {
            "valid": False,
            "threshold": float(threshold),
            "obs_pixels": obs_pixels,
            "pred_pixels": pred_pixels,
            "reason": "missing_observed_object" if obs_pixels == 0 else "missing_forecast_object",
        }

    pred_object = pred_np * pred_mask
    obs_object = target_np * obs_mask
    fixed_mask = obs_mask | pred_mask
    r_original = pearson_corr_np(pred_object[fixed_mask], obs_object[fixed_mask])
    if not np.isfinite(r_original):
        r_original = 0.0

    best = {"r": -float("inf"), "dx": 0, "dy": 0}
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted_pred = shift_2d_zero(pred_object, dx, dy)
            shifted_mask = shift_2d_zero(pred_mask.astype(np.uint8), dx, dy).astype(bool)
            compare_mask = obs_mask | shifted_mask
            r_value = pearson_corr_np(shifted_pred[compare_mask], obs_object[compare_mask])
            if np.isfinite(r_value) and r_value > best["r"]:
                best = {"r": r_value, "dx": dx, "dy": dy}

    if not np.isfinite(best["r"]):
        best = {"r": r_original, "dx": 0, "dy": 0}

    f_values = pred_object[fixed_mask]
    x_values = obs_object[fixed_mask]
    f_mean = float(np.mean(f_values))
    x_mean = float(np.mean(x_values))
    f_std = float(np.std(f_values))
    x_std = float(np.std(x_values))
    mse_total = float((f_mean - x_mean) ** 2 + f_std ** 2 + x_std ** 2 - 2.0 * r_original * f_std * x_std)
    mse_displacement = float(2.0 * f_std * x_std * (best["r"] - r_original))
    mse_volume = float((f_mean - x_mean) ** 2)
    mse_pattern = float(2.0 * f_std * x_std * (1.0 - best["r"]) + (f_std - x_std) ** 2)
    components = {
        "total": max(mse_total, 0.0),
        "displacement": max(mse_displacement, 0.0),
        "volume": max(mse_volume, 0.0),
        "pattern": max(mse_pattern, 0.0),
    }
    total_den = components["total"] if components["total"] > 1e-12 else float("nan")
    return {
        "valid": True,
        "threshold": float(threshold),
        "obs_pixels": obs_pixels,
        "pred_pixels": pred_pixels,
        "dx_pixels": int(best["dx"]),
        "dy_pixels": int(best["dy"]),
        "distance_km": float((best["dx"] ** 2 + best["dy"] ** 2) ** 0.5 * grid_km),
        "direction_deg": float(np.degrees(np.arctan2(best["dy"], best["dx"]))),
        "r_original": float(r_original),
        "r_optimal": float(best["r"]),
        "mse_total": components["total"],
        "mse_displacement": components["displacement"],
        "mse_volume": components["volume"],
        "mse_pattern": components["pattern"],
        "rmse_total": components["total"] ** 0.5,
        "rmse_displacement": components["displacement"] ** 0.5,
        "rmse_volume": components["volume"] ** 0.5,
        "rmse_pattern": components["pattern"] ** 0.5,
        "percent_displacement": safe_divide(components["displacement"] * 100.0, total_den),
        "percent_volume": safe_divide(components["volume"] * 100.0, total_den),
        "percent_pattern": safe_divide(components["pattern"] * 100.0, total_den),
    }


def init_cra_store(thresholds, lead_minutes):
    return {
        str(int(lead)): {threshold_key(thr): [] for thr in thresholds}
        for lead in lead_minutes
    }


def update_cra_store(store, pred, target, frame_minutes, lead_minutes, thresholds, max_shift, grid_km):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for lead in lead_minutes:
        index = int(round(float(lead) / frame_minutes)) - 1
        if index < 0 or index >= pred_np.shape[1]:
            continue
        lead_key = str(int(lead))
        for threshold in thresholds:
            key = threshold_key(threshold)
            for batch_index in range(pred_np.shape[0]):
                store[lead_key][key].append(
                    cra_one_field(
                        pred_np[batch_index, index],
                        target_np[batch_index, index],
                        threshold,
                        max_shift,
                        grid_km,
                    )
                )


def summarize_cra_store(store):
    metric_keys = (
        "distance_km",
        "dx_pixels",
        "dy_pixels",
        "r_original",
        "r_optimal",
        "rmse_total",
        "rmse_displacement",
        "rmse_volume",
        "rmse_pattern",
        "percent_displacement",
        "percent_volume",
        "percent_pattern",
    )
    summary = {}
    for lead, by_threshold in store.items():
        summary[lead] = {}
        for threshold, items in by_threshold.items():
            valid = [item for item in items if item.get("valid")]
            summary[lead][threshold] = {
                "count": len(items),
                "valid_count": len(valid),
                "missing_count": len(items) - len(valid),
                "metrics": {
                    key: nan_summary([item.get(key) for item in valid])
                    for key in metric_keys
                },
            }
    return summary


def main():
    args = add_model_runtime_args(build_parser().parse_args())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_png_dataloader(args, args.split, args.max_samples, shuffle=False, drop_last=False)
    extreme_quantiles = parse_float_list(args.extreme_quantiles)
    intensity_bin_quantiles = parse_float_list(args.intensity_bin_quantiles)
    quantile_info = compute_target_quantile_thresholds(
        loader,
        sorted(set(extreme_quantiles + intensity_bin_quantiles)),
        args.extreme_rain_min,
        args.intensity_scale,
        args.quantile_bins,
    )
    extreme_items = threshold_items_from_quantiles(
        quantile_info, [quantile_label(q) for q in extreme_quantiles]
    )
    fss_items = threshold_items_from_quantiles(
        quantile_info, [quantile_label(q) for q in parse_float_list(args.fss_quantiles)]
    )
    intensity_bins = build_intensity_bins(quantile_info, args.extreme_rain_min)
    fss_neighborhood_sizes = parse_int_list(args.fss_neighborhood_sizes)

    model = build_generator(args)
    model.load_state_dict(load_state(args.checkpoint, args.device))
    model.eval()

    model_totals = init_scalar_totals()
    persistence_totals = init_scalar_totals()
    saved = 0
    thresholds = parse_thresholds(args.metric_thresholds)
    neighborhood_thresholds = parse_thresholds(args.neighborhood_metric_thresholds or args.metric_thresholds)
    horizon_bins = parse_horizon_bins(args.horizon_bins)
    psd_lead_minutes = parse_float_list(args.psd_lead_minutes)
    psd_wavelengths = parse_float_list(args.psd_wavelengths)
    cra_thresholds = parse_thresholds(args.cra_thresholds)
    cra_lead_minutes = parse_float_list(args.cra_lead_minutes)
    model_event_counts = init_event_counts(thresholds)
    persistence_event_counts = init_event_counts(thresholds)
    model_extreme_event_counts = init_labeled_event_counts(extreme_items)
    persistence_extreme_event_counts = init_labeled_event_counts(extreme_items)
    model_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    persistence_neighborhood_counts = init_event_counts(neighborhood_thresholds)
    model_lead_totals = init_lead_totals(args.gen_oc)
    persistence_lead_totals = init_lead_totals(args.gen_oc)
    model_horizon_totals = init_horizon_totals(horizon_bins)
    persistence_horizon_totals = init_horizon_totals(horizon_bins)
    model_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    persistence_intensity_bin_totals = init_intensity_bin_totals(intensity_bins)
    model_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    persistence_fss_totals = init_fss_totals(fss_items, fss_neighborhood_sizes)
    psd_totals = init_psd_totals(psd_lead_minutes, psd_wavelengths)
    model_pearson_totals = init_pearson_totals(args.gen_oc)
    persistence_pearson_totals = init_pearson_totals(args.gen_oc)
    model_eventwise = init_eventwise_store(thresholds)
    persistence_eventwise = init_eventwise_store(thresholds)
    model_cra = init_cra_store(cra_thresholds, cra_lead_minutes)
    persistence_cra = init_cra_store(cra_thresholds, cra_lead_minutes)
    extreme_case_threshold = quantile_info.get("thresholds", {}).get("P99", 0.0)
    extreme_cases = init_extreme_cases(args.num_extreme_cases)

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
            target = batch["target_frames"].float().to(args.device, non_blocking=True)
            pred = model(frames)[..., 0]
            last_input = frames[:, args.input_length - 1, :, :, 0]
            persistence = last_input.unsqueeze(1).repeat(1, args.gen_oc, 1, 1)

            update_scalar_totals(model_totals, pred, target)
            update_scalar_totals(persistence_totals, persistence, target)
            update_lead_and_horizon(model_lead_totals, model_horizon_totals, pred, target, args.frame_minutes, horizon_bins)
            update_lead_and_horizon(persistence_lead_totals, persistence_horizon_totals, persistence, target, args.frame_minutes, horizon_bins)
            update_event_counts(model_event_counts, pred, target, thresholds)
            update_event_counts(persistence_event_counts, persistence, target, thresholds)
            update_labeled_event_counts(model_extreme_event_counts, pred, target, extreme_items)
            update_labeled_event_counts(persistence_extreme_event_counts, persistence, target, extreme_items)
            update_neighborhood_event_counts(model_neighborhood_counts, pred, target, neighborhood_thresholds, args.neighborhood_size)
            update_neighborhood_event_counts(persistence_neighborhood_counts, persistence, target, neighborhood_thresholds, args.neighborhood_size)
            update_intensity_bin_totals(model_intensity_bin_totals, pred, target, intensity_bins)
            update_intensity_bin_totals(persistence_intensity_bin_totals, persistence, target, intensity_bins)
            update_fss_totals(model_fss_totals, pred, target, fss_items, fss_neighborhood_sizes)
            update_fss_totals(persistence_fss_totals, persistence, target, fss_items, fss_neighborhood_sizes)
            update_psd_totals(psd_totals, pred, target, persistence, args.frame_minutes, psd_lead_minutes, psd_wavelengths, args.grid_km)
            update_pearson_totals(model_pearson_totals, pred, target)
            update_pearson_totals(persistence_pearson_totals, persistence, target)
            update_eventwise_store(model_eventwise, pred, target, thresholds)
            update_eventwise_store(persistence_eventwise, persistence, target, thresholds)
            update_cra_store(model_cra, pred, target, args.frame_minutes, cra_lead_minutes, cra_thresholds, args.cra_max_shift, args.grid_km)
            update_cra_store(persistence_cra, persistence, target, args.frame_minutes, cra_lead_minutes, cra_thresholds, args.cra_max_shift, args.grid_km)

            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            input_np = frames.detach().cpu().numpy()[..., 0]
            update_extreme_cases(
                extreme_cases,
                args.num_extreme_cases,
                batch,
                pred,
                target,
                persistence,
                {"input": input_np[:, :args.input_length]},
                extreme_case_threshold,
            )

            for i in range(pred_np.shape[0]):
                if saved >= args.num_save_samples:
                    continue
                sample_dir = output_dir / "sample_{:04d}".format(saved)
                save_sequence(sample_dir, "input_", input_np[i, :args.input_length], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                save_sequence(sample_dir, "gt_", target_np[i], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                save_sequence(sample_dir, "pd_", pred_np[i], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                save_sequence(sample_dir, "ps_", persistence.detach().cpu().numpy()[i], args.intensity_scale, args.pixel_min, args.pixel_max, not args.no_invert)
                saved += 1

            print("tested batch {}".format(batch_id + 1), flush=True)

    model_neighborhood_metrics = finalize_neighborhood_metrics(model_neighborhood_counts)
    persistence_neighborhood_metrics = finalize_neighborhood_metrics(persistence_neighborhood_counts)
    save_extreme_cases(output_dir, extreme_cases, quantile_info.get("thresholds", {}), args, not args.no_invert)
    metrics = {
        "model": finalize_scalar_totals(model_totals),
        "persistence": finalize_scalar_totals(persistence_totals),
        "samples": len(loader.dataset),
        "saved_samples": saved,
        "units": {
            "prediction": "mm/h",
            "thresholds": "mm/h",
            "pixel_mapping": "255->0, 0->intensity_scale",
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
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
