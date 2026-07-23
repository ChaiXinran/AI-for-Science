"""Test whether observed PWV adds information after frozen radar history.

The diagnostic freezes a trained NowcastNet evolution path and generative
encoder, then trains matched-capacity event probes on its 1/8-resolution radar
latent.  The PWV model is evaluated with aligned, spatially shifted, and
cross-event PWV using the same checkpoint.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nowcasting.experiments.common import (  # noqa: E402
    add_model_runtime_args,
    build_generator,
    load_generator_weights,
    load_model_state,
    make_png_dataloader,
    safe_torch_save,
    save_dataset_provenance,
    seed_everything,
)
from nowcasting.layers.utils import warp  # noqa: E402


HORIZONS = (("0h-1h", 0, 10), ("1h-2h", 10, 20))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Frozen-radar conditional information probe for PWV"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pwv_root", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--radar_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", default="NowcastNet")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--lead_time_embed_dim", type=int, default=16)
    parser.add_argument("--intensity_scale", type=float, default=35.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--pwv_intensity_scale", type=float, default=80.0)
    parser.add_argument("--pwv_pixel_min", type=float, default=0.0)
    parser.add_argument("--pwv_pixel_max", type=float, default=255.0)
    parser.add_argument("--pwv_invert", action="store_true")
    parser.add_argument("--strict_pwv", action="store_true")
    parser.add_argument("--require_contiguous", action="store_true")
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=2048)
    parser.add_argument("--max_val_samples", type=int, default=512)
    parser.add_argument(
        "--max_samples_strategy", choices=["head", "uniform"], default="uniform"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--calibration_day_fraction", type=float, default=0.2)
    parser.add_argument("--thresholds", default="10,20")
    parser.add_argument("--histogram_bins", type=int, default=1000)
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--minimum_csi_delta", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reuse_cache", action="store_true")
    return parser


class ConditionalEventProbe(nn.Module):
    """Matched architecture with a real or learned-constant auxiliary slot."""

    def __init__(
        self,
        radar_channels,
        pwv_channels,
        hidden_channels,
        lead_count,
        threshold_count,
    ):
        super().__init__()
        self.radar_encoder = nn.Sequential(
            nn.Conv2d(radar_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
        )
        self.pwv_encoder = nn.Sequential(
            nn.Conv2d(pwv_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
        )
        self.null_auxiliary = nn.Parameter(
            torch.zeros(1, hidden_channels, 1, 1)
        )
        self.lead_embedding = nn.Parameter(
            torch.zeros(lead_count, hidden_channels)
        )
        nn.init.trunc_normal_(self.lead_embedding, std=0.02)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, threshold_count, 1),
        )
        self.lead_count = lead_count

    def forward(self, radar_latent, pwv_features, use_pwv):
        radar = self.radar_encoder(radar_latent)
        if use_pwv:
            auxiliary = self.pwv_encoder(pwv_features)
        else:
            auxiliary = self.null_auxiliary.expand(
                radar.size(0), -1, radar.size(2), radar.size(3)
            )
        outputs = []
        for lead in range(self.lead_count):
            lead_feature = self.lead_embedding[lead].view(1, -1, 1, 1)
            outputs.append(
                self.fusion(
                    torch.cat([radar + lead_feature, auxiliary], dim=1)
                )
            )
        return torch.stack(outputs, dim=1)


def _frozen_radar_latent(model, frames, args):
    frames = frames[..., 0]
    input_frames = frames[:, : args.input_length]
    batch, _, height, width = input_frames.shape
    intensity, motion = model.evo_net(input_frames)
    intensity = model.lead_time(intensity, "source")
    motion = motion.reshape(batch, args.evo_ic, 2, height, width)
    motion = model.lead_time(motion, "motion")
    intensity = intensity.reshape(batch, args.evo_ic, 1, height, width)
    last = input_frames[:, -1:]
    grid = model.grid.to(frames.device).repeat(batch, 1, 1, 1)
    series = []
    for lead in range(args.evo_ic):
        last = warp(
            last,
            motion[:, lead],
            grid,
            mode="nearest",
            padding_mode="border",
        ) + intensity[:, lead]
        series.append(last)
    evolution = torch.cat(series, dim=1) / max(args.intensity_scale, 1e-6)
    return model.gen_enc(torch.cat([input_frames, evolution], dim=1))


def _pwv_features(pwv, args, latent_size):
    observed = pwv[:, : args.input_length]
    scale = max(args.pwv_intensity_scale, 1e-6)
    last = observed[:, -1:]
    mean = observed.mean(dim=1, keepdim=True)
    std = observed.std(dim=1, keepdim=True)
    tendency = observed[:, -1:] - observed[:, :1]
    spatial_anomaly = last - last.mean(dim=(-2, -1), keepdim=True)
    dx = F.pad(last[..., 1:] - last[..., :-1], (0, 1, 0, 0))
    dy = F.pad(last[..., 1:, :] - last[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx.square() + dy.square() + 1e-8)
    features = torch.cat(
        [last, mean, std, tendency, spatial_anomaly, gradient], dim=1
    ) / scale
    return F.adaptive_avg_pool2d(features, latent_size).clamp(-2.0, 2.0)


def _event_targets(target, thresholds, latent_size):
    outputs = []
    for threshold in thresholds:
        binary = (target >= threshold).float()
        batch, leads, height, width = binary.shape
        pooled = F.adaptive_max_pool2d(
            binary.reshape(batch * leads, 1, height, width), latent_size
        )
        outputs.append(
            pooled.reshape(batch, leads, latent_size[0], latent_size[1])
        )
    return torch.stack(outputs, dim=2).to(torch.uint8)


@torch.no_grad()
def build_cache(model, loader, args, thresholds):
    model.eval()
    radar_parts = []
    pwv_parts = []
    target_parts = []
    cases = []
    sample_ids = []
    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device)
        pwv = batch["pwv_frames"].float().to(args.device)
        target = batch["target_frames"].float().to(args.device)
        radar = _frozen_radar_latent(model, frames, args)
        latent_size = radar.shape[-2:]
        radar_parts.append(radar.cpu().half())
        pwv_parts.append(_pwv_features(pwv, args, latent_size).cpu().half())
        target_parts.append(
            _event_targets(target, thresholds, latent_size).cpu()
        )
        cases.extend(list(batch["case_name"]))
        sample_ids.extend(list(batch["sample_id"]))
        if step % 20 == 0:
            print("cached batches {}".format(step), flush=True)
    return {
        "radar": torch.cat(radar_parts),
        "pwv": torch.cat(pwv_parts),
        "target": torch.cat(target_parts),
        "cases": cases,
        "sample_ids": sample_ids,
    }


def split_fit_calibration(cases, fraction):
    unique = sorted(set(cases))
    if len(unique) < 2:
        indices = torch.arange(len(cases))
        return indices, indices
    calibration_days = max(1, int(round(len(unique) * fraction)))
    calibration_set = set(unique[-calibration_days:])
    fit = [i for i, case in enumerate(cases) if case not in calibration_set]
    calibration = [
        i for i, case in enumerate(cases) if case in calibration_set
    ]
    if not fit:
        raise ValueError("Calibration split consumed every training case.")
    return torch.tensor(fit), torch.tensor(calibration)


def _batch_indices(indices, batch_size, shuffle, seed):
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        order = indices[torch.randperm(len(indices), generator=generator)]
    else:
        order = indices
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def cross_event_indices(cases):
    """Map every sample to a deterministic sample from a different case."""
    by_case = defaultdict(list)
    for index, case in enumerate(cases):
        by_case[case].append(index)
    unique = sorted(by_case)
    if len(unique) < 2:
        return torch.roll(torch.arange(len(cases)), shifts=max(1, len(cases) // 2))
    mapping = torch.empty(len(cases), dtype=torch.long)
    for case_index, case in enumerate(unique):
        donor = unique[(case_index + max(1, len(unique) // 2)) % len(unique)]
        donor_indices = by_case[donor]
        for position, sample_index in enumerate(by_case[case]):
            mapping[sample_index] = donor_indices[position % len(donor_indices)]
    return mapping


def positive_weights(target, indices):
    selected = target[indices].float()
    positives = selected.sum(dim=(0, 1, 3, 4))
    total = selected.shape[0] * selected.shape[1] * selected.shape[3] * selected.shape[4]
    negatives = total - positives
    return (negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0)


def train_probe(model, cache, fit_indices, use_pwv, args, label):
    model.to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    pos_weight = positive_weights(cache["target"], fit_indices).to(args.device)
    pos_weight = pos_weight.view(1, 1, -1, 1, 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for indices in _batch_indices(
            fit_indices,
            args.probe_batch_size,
            True,
            args.seed + epoch,
        ):
            radar = cache["radar"][indices].float().to(args.device)
            pwv = cache["pwv"][indices].float().to(args.device)
            target = cache["target"][indices].float().to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(radar, pwv, use_pwv=use_pwv)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            total_samples += len(indices)
        print(
            "{} epoch {:03d} loss {:.6f}".format(
                label, epoch, total_loss / max(total_samples, 1)
            ),
            flush=True,
        )
    return model.cpu()


def _histogram(probability, target, bins):
    index = torch.clamp((probability * bins).long(), 0, bins - 1)
    positive = torch.bincount(
        index[target], minlength=bins
    ).double()
    negative = torch.bincount(
        index[~target], minlength=bins
    ).double()
    return positive, negative


def _curve_from_histogram(positive, negative):
    tp_descending = torch.cumsum(torch.flip(positive, dims=(0,)), dim=0)
    fp_descending = torch.cumsum(torch.flip(negative, dims=(0,)), dim=0)
    tp = torch.flip(tp_descending, dims=(0,))
    fp = torch.flip(fp_descending, dims=(0,))
    total_positive = positive.sum()
    fn = total_positive - tp
    denominator = tp + fp + fn
    csi = torch.where(denominator > 0, tp / denominator, torch.zeros_like(tp))
    precision_descending = torch.where(
        tp_descending + fp_descending > 0,
        tp_descending / (tp_descending + fp_descending),
        torch.ones_like(tp_descending),
    )
    recall_descending = torch.where(
        total_positive > 0,
        tp_descending / total_positive,
        torch.zeros_like(tp_descending),
    )
    recall_previous = torch.cat(
        [torch.zeros(1, dtype=recall_descending.dtype), recall_descending[:-1]]
    )
    average_precision = (
        precision_descending * (recall_descending - recall_previous)
    ).sum()
    precision = torch.flip(precision_descending, dims=(0,))
    recall = torch.flip(recall_descending, dims=(0,))
    return csi, precision, recall, average_precision


@torch.no_grad()
def predict(model, cache, use_pwv, args, control="real"):
    model.to(args.device).eval()
    outputs = []
    count = len(cache["radar"])
    cross_event = cross_event_indices(cache["cases"])
    all_indices = torch.arange(count)
    for indices in _batch_indices(
        all_indices, args.probe_batch_size, False, args.seed
    ):
        radar = cache["radar"][indices].float().to(args.device)
        if control == "cross_event":
            pwv = cache["pwv"][cross_event[indices]].float().to(args.device)
        else:
            pwv = cache["pwv"][indices].float().to(args.device)
        if control == "spatial_shift":
            pwv = torch.roll(
                pwv,
                shifts=(max(1, pwv.shape[-2] // 2), max(1, pwv.shape[-1] // 2)),
                dims=(-2, -1),
            )
        outputs.append(
            torch.sigmoid(model(radar, pwv, use_pwv=use_pwv)).cpu()
        )
    return torch.cat(outputs)


def calibrate_thresholds(model, cache, indices, use_pwv, thresholds, args):
    subset = {
        key: value[indices] if torch.is_tensor(value) else [value[i] for i in indices]
        for key, value in cache.items()
    }
    probabilities = predict(model, subset, use_pwv, args)
    target = subset["target"].bool()
    selected = {}
    for horizon, start, end in HORIZONS:
        selected[horizon] = {}
        for threshold_index, threshold in enumerate(thresholds):
            positive, negative = _histogram(
                probabilities[:, start:end, threshold_index].reshape(-1),
                target[:, start:end, threshold_index].reshape(-1),
                args.histogram_bins,
            )
            csi, _, _, _ = _curve_from_histogram(positive, negative)
            best = int(torch.argmax(csi).item())
            selected[horizon]["{:g}".format(threshold)] = {
                "probability_threshold": best / args.histogram_bins,
                "calibration_csi": float(csi[best]),
            }
    return selected


def event_metrics(probability, target, probability_threshold):
    prediction = probability >= probability_threshold
    hit = int((prediction & target).sum())
    miss = int((~prediction & target).sum())
    false_alarm = int((prediction & ~target).sum())
    correct_negative = int((~prediction & ~target).sum())
    event_denominator = hit + miss + false_alarm
    return {
        "hit": hit,
        "miss": miss,
        "false_alarm": false_alarm,
        "correct_negative": correct_negative,
        "csi": hit / event_denominator if event_denominator else None,
        "pod": hit / (hit + miss) if hit + miss else None,
        "far": false_alarm / (hit + false_alarm) if hit + false_alarm else None,
    }


def evaluate_variant(
    model,
    cache,
    use_pwv,
    control,
    selected_thresholds,
    thresholds,
    args,
):
    probabilities = predict(model, cache, use_pwv, args, control=control)
    target = cache["target"].bool()
    summary = {"control": control, "horizons": {}, "eventwise": []}
    per_sample = defaultdict(dict)
    for horizon, start, end in HORIZONS:
        summary["horizons"][horizon] = {}
        for threshold_index, threshold in enumerate(thresholds):
            key = "{:g}".format(threshold)
            probability = probabilities[:, start:end, threshold_index]
            truth = target[:, start:end, threshold_index]
            cutoff = selected_thresholds[horizon][key][
                "probability_threshold"
            ]
            metrics = event_metrics(probability, truth, cutoff)
            positive, negative = _histogram(
                probability.reshape(-1), truth.reshape(-1), args.histogram_bins
            )
            _, _, _, ap = _curve_from_histogram(positive, negative)
            metrics["average_precision"] = float(ap)
            metrics["probability_threshold"] = cutoff
            summary["horizons"][horizon][key] = metrics
            for sample in range(len(cache["sample_ids"])):
                per_sample[sample]["{}::{}".format(horizon, key)] = event_metrics(
                    probability[sample], truth[sample], cutoff
                )
    for sample, sample_id in enumerate(cache["sample_ids"]):
        summary["eventwise"].append(
            {
                "sample_id": sample_id,
                "case_name": cache["cases"][sample],
                "events": per_sample[sample],
            }
        )
    return summary


def _aggregate_eventwise(rows, event_key):
    hit = miss = false_alarm = 0
    for row in rows:
        event = row["events"][event_key]
        hit += event["hit"]
        miss += event["miss"]
        false_alarm += event["false_alarm"]
    denominator = hit + miss + false_alarm
    return hit / denominator if denominator else None


def paired_day_bootstrap(left, right, event_key, repetitions, seed):
    left_map = {row["sample_id"]: row for row in left}
    right_map = {row["sample_id"]: row for row in right}
    if set(left_map) != set(right_map):
        raise ValueError("Paired variants have different sample IDs.")
    by_case = defaultdict(list)
    for sample_id in sorted(left_map):
        left_row, right_row = left_map[sample_id], right_map[sample_id]
        if left_row["case_name"] != right_row["case_name"]:
            raise ValueError("Paired variants disagree on case name.")
        by_case[left_row["case_name"]].append((left_row, right_row))
    cases = sorted(by_case)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        selected = rng.choice(cases, size=len(cases), replace=True)
        sampled_left, sampled_right = [], []
        for case in selected:
            for left_row, right_row in by_case[case]:
                sampled_left.append(left_row)
                sampled_right.append(right_row)
        left_csi = _aggregate_eventwise(sampled_left, event_key)
        right_csi = _aggregate_eventwise(sampled_right, event_key)
        if left_csi is not None and right_csi is not None:
            deltas.append(left_csi - right_csi)
    values = np.asarray(deltas)
    return {
        "n_cases": len(cases),
        "repetitions": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "ci95": (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if len(values)
            else [None, None]
        ),
    }


def optional_delta(left, right):
    if left is None or right is None:
        return None
    return left - right


def main():
    args = build_parser().parse_args()
    args = add_model_runtime_args(args)
    thresholds = [float(item) for item in args.thresholds.split(",")]
    if len(thresholds) != 2:
        raise ValueError("This locked probe requires exactly two thresholds.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    train_loader = make_png_dataloader(
        args, "train", args.max_train_samples, shuffle=False, drop_last=False
    )
    val_loader = make_png_dataloader(
        args, "val", args.max_val_samples, shuffle=False, drop_last=False
    )
    save_dataset_provenance(
        {"train": train_loader, "val": val_loader},
        output / "data_manifest.json",
    )
    cache_path = output / "feature_cache.pt"
    if args.reuse_cache and cache_path.exists():
        caches = torch.load(cache_path, map_location="cpu")
    else:
        radar = build_generator(args)
        load_generator_weights(
            radar, args.radar_checkpoint, args.device, strict=True
        )
        radar.requires_grad_(False)
        caches = {
            "train": build_cache(radar, train_loader, args, thresholds),
            "val": build_cache(radar, val_loader, args, thresholds),
        }
        safe_torch_save(caches, cache_path)
        del radar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fit_indices, calibration_indices = split_fit_calibration(
        caches["train"]["cases"], args.calibration_day_fraction
    )
    radar_channels = caches["train"]["radar"].shape[1]
    pwv_channels = caches["train"]["pwv"].shape[1]
    probe_kwargs = dict(
        radar_channels=radar_channels,
        pwv_channels=pwv_channels,
        hidden_channels=args.hidden_channels,
        lead_count=args.evo_ic,
        threshold_count=len(thresholds),
    )
    seed_everything(args.seed)
    radar_probe = ConditionalEventProbe(**probe_kwargs)
    seed_everything(args.seed)
    aligned_probe = ConditionalEventProbe(**probe_kwargs)
    if sum(p.numel() for p in radar_probe.parameters()) != sum(
        p.numel() for p in aligned_probe.parameters()
    ):
        raise AssertionError("Probe parameter counts must match.")

    radar_probe = train_probe(
        radar_probe, caches["train"], fit_indices, False, args, "radar"
    )
    aligned_probe = train_probe(
        aligned_probe, caches["train"], fit_indices, True, args, "aligned"
    )
    radar_cutoffs = calibrate_thresholds(
        radar_probe,
        caches["train"],
        calibration_indices,
        False,
        thresholds,
        args,
    )
    aligned_cutoffs = calibrate_thresholds(
        aligned_probe,
        caches["train"],
        calibration_indices,
        True,
        thresholds,
        args,
    )
    variants = {
        "radar_only": evaluate_variant(
            radar_probe,
            caches["val"],
            False,
            "learned_constant",
            radar_cutoffs,
            thresholds,
            args,
        ),
        "pwv_aligned": evaluate_variant(
            aligned_probe,
            caches["val"],
            True,
            "real",
            aligned_cutoffs,
            thresholds,
            args,
        ),
        "pwv_spatial_shift": evaluate_variant(
            aligned_probe,
            caches["val"],
            True,
            "spatial_shift",
            aligned_cutoffs,
            thresholds,
            args,
        ),
        "pwv_cross_event": evaluate_variant(
            aligned_probe,
            caches["val"],
            True,
            "cross_event",
            aligned_cutoffs,
            thresholds,
            args,
        ),
    }
    comparisons = {}
    for reference in ("radar_only", "pwv_spatial_shift", "pwv_cross_event"):
        comparison = {}
        for horizon, _, _ in HORIZONS:
            comparison[horizon] = {}
            for threshold in thresholds:
                key = "{:g}".format(threshold)
                event_key = "{}::{}".format(horizon, key)
                left = variants["pwv_aligned"]["horizons"][horizon][key]
                right = variants[reference]["horizons"][horizon][key]
                comparison[horizon][key] = {
                    "csi_delta": optional_delta(left["csi"], right["csi"]),
                    "average_precision_delta": optional_delta(
                        left["average_precision"],
                        right["average_precision"],
                    ),
                    "day_cluster_bootstrap_csi_delta": paired_day_bootstrap(
                        variants["pwv_aligned"]["eventwise"],
                        variants[reference]["eventwise"],
                        event_key,
                        args.bootstrap_repetitions,
                        args.seed,
                    ),
                }
        comparisons["pwv_aligned_minus_" + reference] = comparison
    task_gate = {}
    passing_tasks = 0
    safety_pass = True
    for horizon, _, _ in HORIZONS:
        for threshold in thresholds:
            key = "{:g}".format(threshold)
            task_name = "{}::{}".format(horizon, key)
            reference_results = {}
            task_pass = True
            for reference in (
                "radar_only",
                "pwv_spatial_shift",
                "pwv_cross_event",
            ):
                item = comparisons["pwv_aligned_minus_" + reference][horizon][key]
                csi_delta = item["csi_delta"]
                ap_delta = item["average_precision_delta"]
                reference_pass = (
                    csi_delta is not None
                    and csi_delta >= args.minimum_csi_delta
                    and ap_delta is not None
                    and ap_delta > 0
                )
                if csi_delta is not None and csi_delta < -args.minimum_csi_delta:
                    safety_pass = False
                reference_results[reference] = reference_pass
                task_pass = task_pass and reference_pass
            task_gate[task_name] = {
                "pass": task_pass,
                "references": reference_results,
            }
            passing_tasks += int(task_pass)
    for variant in variants.values():
        variant.pop("eventwise")
    summary = {
        "protocol": "pwv_conditional_information_probe",
        "tile_size_pixels": args.img_height // caches["val"]["radar"].shape[-2],
        "train_samples": len(caches["train"]["radar"]),
        "validation_samples": len(caches["val"]["radar"]),
        "fit_samples": len(fit_indices),
        "calibration_samples": len(calibration_indices),
        "probe_parameters": sum(p.numel() for p in aligned_probe.parameters()),
        "frozen_radar_parameters": sum(
            tensor.numel()
            for tensor in load_model_state(
                args.radar_checkpoint, "cpu"
            ).values()
            if torch.is_tensor(tensor)
        ),
        "thresholds_mm_h": thresholds,
        "calibration_thresholds": {
            "radar_only": radar_cutoffs,
            "pwv_aligned": aligned_cutoffs,
        },
        "variants": variants,
        "paired_comparisons": comparisons,
        "gate": {
            "minimum_csi_delta": args.minimum_csi_delta,
            "required_passing_tasks": 3,
            "passing_tasks": passing_tasks,
            "safety_pass": safety_pass,
            "point_estimate_pass": passing_tasks >= 3 and safety_pass,
            "tasks": task_gate,
        },
        "interpretation": (
            "This is a conditional-information diagnostic, not a final dense "
            "forecast. Positive evidence requires aligned PWV to beat radar "
            "and both same-checkpoint PWV interventions."
        ),
    }
    (output / "conditional_probe_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    safe_torch_save(radar_probe.state_dict(), output / "radar_probe.ckpt")
    safe_torch_save(aligned_probe.state_dict(), output / "pwv_probe.ckpt")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
