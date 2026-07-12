import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
    import cv2
except ImportError:
    cv2 = None

from nowcasting.data_provider.custom_png import PngSequenceDataset
from nowcasting.models import nowcastnet


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
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--horizon_bins", type=str, default="0-1,1-2,2-3,3-6")
    parser.add_argument("--no_invert", action="store_true")
    return parser


def load_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


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
        hss_den = (hit + miss) * (miss + correct_negative) + (hit + false_alarm) * (false_alarm + correct_negative)
        metrics[threshold] = {
            "hit": hit,
            "miss": miss,
            "false_alarm": false_alarm,
            "correct_negative": correct_negative,
            "csi": hit / csi_den if csi_den else 0.0,
            "pod": hit / pod_den if pod_den else 0.0,
            "far": false_alarm / far_den if far_den else 0.0,
            "bias": (hit + false_alarm) / bias_den if bias_den else 0.0,
            "hss": (2 * (hit * correct_negative - false_alarm * miss) / hss_den) if hss_den else 0.0,
        }
    return metrics


def init_scalar_totals():
    return {"abs": 0.0, "sq": 0.0, "count": 0}


def update_scalar_totals(totals, pred, target):
    diff = pred - target
    totals["abs"] += diff.abs().sum().item()
    totals["sq"] += (diff ** 2).sum().item()
    totals["count"] += diff.numel()


def finalize_scalar_totals(totals):
    count = max(totals["count"], 1)
    mse = totals["sq"] / count
    return {
        "mae": totals["abs"] / count,
        "mse": mse,
        "rmse": mse ** 0.5,
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


def main():
    args = build_parser().parse_args()
    args.evo_ic = args.total_length - args.input_length
    args.gen_oc = args.total_length - args.input_length
    args.ic_feature = args.ngf * 10

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PngSequenceDataset(
        data_root=args.data_root,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        input_length=args.input_length,
        total_length=args.total_length,
        img_height=args.img_height,
        img_width=args.img_width,
        stride=args.stride,
        max_samples=args.max_samples,
        intensity_scale=args.intensity_scale,
        pixel_min=args.pixel_min,
        pixel_max=args.pixel_max,
        invert=not args.no_invert,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = nowcastnet.Net(args).to(args.device)
    model.load_state_dict(load_state(args.checkpoint, args.device))
    model.eval()

    model_totals = init_scalar_totals()
    persistence_totals = init_scalar_totals()
    saved = 0
    thresholds = parse_thresholds(args.metric_thresholds)
    horizon_bins = parse_horizon_bins(args.horizon_bins)
    model_event_counts = init_event_counts(thresholds)
    persistence_event_counts = init_event_counts(thresholds)
    model_lead_totals = init_lead_totals(args.gen_oc)
    persistence_lead_totals = init_lead_totals(args.gen_oc)
    model_horizon_totals = init_horizon_totals(horizon_bins)
    persistence_horizon_totals = init_horizon_totals(horizon_bins)

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

            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            input_np = frames.detach().cpu().numpy()[..., 0]

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

    metrics = {
        "model": finalize_scalar_totals(model_totals),
        "persistence": finalize_scalar_totals(persistence_totals),
        "samples": len(dataset),
        "saved_samples": saved,
        "thresholds": thresholds,
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
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
