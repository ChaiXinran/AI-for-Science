"""Train the frozen-radar signed PWV calibrator.

This intentionally avoids adversarial and positive-source losses.  The small
head is optimized for 10/20 mm/h occurrence skill while a preservation term
limits broad changes to the frozen radar forecast.
"""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from nowcasting.birth_growth import apply_pwv_control
from nowcasting.experiments.common import (
    add_model_runtime_args,
    build_generator,
    load_radar_backbone_weights,
    make_png_dataloader,
    save_dataset_provenance,
    save_json_args,
)
from train.pwv import build_parser as build_shared_parser
from train.radar import safe_torch_save, seed_everything, weighted_l1


def parse_thresholds(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def balanced_threshold_loss(prediction, target, thresholds, temperature):
    losses = []
    parts = {}
    temperature = max(float(temperature), 1e-3)
    for threshold in thresholds:
        logits = (prediction - threshold) / temperature
        labels = (target >= threshold).float()
        raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        positive = labels > 0.5
        negative = ~positive
        zero = raw.sum() * 0.0
        positive_loss = raw[positive].mean() if positive.any() else zero
        negative_loss = raw[negative].mean() if negative.any() else zero
        loss = 0.5 * positive_loss + 0.5 * negative_loss
        losses.append(loss)
        parts["threshold_{:g}".format(threshold)] = loss.detach()
    return torch.stack(losses).mean(), parts


def false_alarm_probability_loss(prediction, target, thresholds, temperature):
    losses = []
    temperature = max(float(temperature), 1e-3)
    for threshold in thresholds:
        probability = torch.sigmoid((prediction - threshold) / temperature)
        negative = target < threshold
        if negative.any():
            losses.append(probability[negative].mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def apply_training_control(pwv, mode, input_length):
    if mode == "batch_shuffle":
        if pwv.size(0) <= 1:
            return torch.flip(pwv, dims=[1])
        return pwv[torch.randperm(pwv.size(0), device=pwv.device)]
    return apply_pwv_control(pwv, mode, input_length)


def hard_event_counts(prediction, target, thresholds):
    counts = {}
    for threshold in thresholds:
        predicted = prediction >= threshold
        observed = target >= threshold
        counts["{:g}".format(threshold)] = {
            "tp": int((predicted & observed).sum()),
            "fp": int((predicted & ~observed).sum()),
            "fn": int((~predicted & observed).sum()),
        }
    return counts


def merge_counts(total, update):
    for threshold, row in update.items():
        target = total.setdefault(threshold, {"tp": 0, "fp": 0, "fn": 0})
        for key in target:
            target[key] += int(row[key])


def finalize_counts(counts):
    result = {}
    for threshold, row in counts.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        result[threshold] = {
            **row,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else None,
            "pod": tp / (tp + fn) if tp + fn else None,
            "far": fp / (tp + fp) if tp + fp else None,
        }
    return result


def compute_loss(generator, frames, pwv, target, args):
    aux = generator(frames, pwv, return_aux=True)
    prediction = aux["prediction"][..., 0]
    thresholds = parse_thresholds(args.calibration_thresholds)
    threshold_loss, threshold_parts = balanced_threshold_loss(
        prediction, target, thresholds, args.calibration_temperature
    )
    forecast_loss = weighted_l1(prediction, target, args.intensity_scale)
    false_alarm = false_alarm_probability_loss(
        prediction, target, thresholds, args.calibration_temperature
    )
    contribution = aux["pwv_contribution"].abs().mean()
    total = (
        args.lambda_calibration * threshold_loss
        + args.lambda_forecast * forecast_loss
        + args.lambda_false_alarm * false_alarm
        + args.lambda_signed_contribution * contribution
    )
    parts = {
        "total": total.detach(),
        "calibration": threshold_loss.detach(),
        "forecast": forecast_loss.detach(),
        "false_alarm": false_alarm.detach(),
        "signed_contribution": contribution.detach(),
        **threshold_parts,
    }
    return total, parts, prediction


def train_one_epoch(generator, loader, optimizer, args):
    generator.train()
    totals = {}
    seen = 0
    for batch_id, batch in enumerate(loader):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        pwv = apply_training_control(pwv, args.pwv_control, args.input_length)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, parts, _ = compute_loss(generator, frames, pwv, target, args)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite signed calibrator loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in generator.parameters() if parameter.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        batch_size = frames.size(0)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        seen += batch_size
        if (batch_id + 1) % args.log_interval == 0:
            print(
                "step {:05d} {}".format(
                    batch_id + 1,
                    " ".join(
                        "{} {:.5f}".format(key, totals[key] / seen)
                        for key in sorted(totals)
                    ),
                ),
                flush=True,
            )
    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(generator, loader, args):
    generator.eval()
    totals = {}
    counts = {}
    seen = 0
    thresholds = parse_thresholds(args.calibration_thresholds)
    for batch_id, batch in enumerate(loader):
        frames = batch["radar_frames"].float().to(args.device, non_blocking=True)
        pwv = batch["pwv_frames"].float().to(args.device, non_blocking=True)
        pwv = apply_training_control(pwv, args.pwv_control, args.input_length)
        target = batch["target_frames"].float().to(args.device, non_blocking=True)
        torch.manual_seed(args.seed + batch_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + batch_id)
        _, parts, prediction = compute_loss(generator, frames, pwv, target, args)
        batch_size = frames.size(0)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        merge_counts(counts, hard_event_counts(prediction, target, thresholds))
        seen += batch_size
    averages = {key: value / max(seen, 1) for key, value in totals.items()}
    return averages, finalize_counts(counts)


def append_log(path, epoch, train_metrics, val_metrics, event_metrics):
    row = {"epoch": epoch}
    row.update({"train_{}".format(key): value for key, value in train_metrics.items()})
    row.update({"val_{}".format(key): value for key, value in val_metrics.items()})
    for threshold, metrics in event_metrics.items():
        for key in ("csi", "pod", "far"):
            row["val_{}_{}".format(key, threshold)] = metrics[key]
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_parser():
    parser = build_shared_parser()
    parser.description = "Train a frozen-radar bounded signed PWV calibrator"
    parser.set_defaults(
        model_name="PWVSignedCalibratorNowcastNet",
        lambda_forecast=0.10,
        lambda_false_alarm=0.10,
        pwv_candidate_radius=4,
        pwv_tendency_windows="",
    )
    parser.add_argument("--pwv_climatology_path", required=True)
    parser.add_argument("--signed_use_tendency", action="store_true")
    parser.add_argument("--signed_residual_scale", type=float, default=0.25)
    parser.add_argument("--calibration_thresholds", default="10,20")
    parser.add_argument("--calibration_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_calibration", type=float, default=1.0)
    parser.add_argument("--lambda_signed_contribution", type=float, default=0.02)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    return parser


def main():
    args = add_model_runtime_args(build_parser().parse_args())
    if args.model_name != "PWVSignedCalibratorNowcastNet":
        raise ValueError("This trainer only supports PWVSignedCalibratorNowcastNet.")
    if not args.init_radar_checkpoint:
        raise ValueError("--init_radar_checkpoint is required.")
    if args.pwv_control == "zero":
        raise ValueError("A zero-PWV head has no trainable signal; evaluate zero on the real checkpoint.")
    seed_everything(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json_args(args, save_dir)

    train_loader = make_png_dataloader(args, "train", args.max_train_samples)
    val_loader = make_png_dataloader(args, "val", args.max_val_samples)
    save_dataset_provenance(
        {"train": train_loader, "val": val_loader},
        save_dir / "data_manifest.json",
    )
    print(
        "train windows: {} val windows: {}".format(
            len(train_loader.dataset), len(val_loader.dataset)
        ),
        flush=True,
    )

    generator = build_generator(args)
    report = load_radar_backbone_weights(
        generator, args.init_radar_checkpoint, args.device
    )
    generator.freeze_radar_backbone()
    trainable = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    print(
        "initialized frozen radar backbone {} trainable_params={}".format(
            report, sum(parameter.numel() for parameter in trainable)
        ),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr_g, betas=(args.beta1, args.beta2), weight_decay=1e-4
    )

    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(generator, train_loader, optimizer, args)
        val_metrics, event_metrics = validate(generator, val_loader, args)
        append_log(
            save_dir / "train_log.csv",
            epoch,
            train_metrics,
            val_metrics,
            event_metrics,
        )
        print(
            "epoch {:03d} val_total {:.6f} events {}".format(
                epoch, val_metrics["total"], json.dumps(event_metrics, sort_keys=True)
            ),
            flush=True,
        )
        safe_torch_save(generator.state_dict(), save_dir / "latest_state_dict.ckpt")
        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]
            stale_epochs = 0
            safe_torch_save(generator.state_dict(), save_dir / "best_state_dict.ckpt")
            safe_torch_save(generator.state_dict(), args.readme_ckpt)
            (save_dir / "best_validation.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "selection_loss": best_loss,
                        "losses": val_metrics,
                        "event_metrics": event_metrics,
                    },
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print("early stopping after {} stale epochs".format(stale_epochs), flush=True)
                break


if __name__ == "__main__":
    main()
