"""Matched radar versus PWV output-adapter pilot for 0--2 h nowcasting."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nowcasting.experiments.common import (  # noqa: E402
    add_model_runtime_args,
    build_generator,
    load_generator_weights,
    make_png_dataloader,
    safe_torch_save,
    sanitize_json_numbers,
    save_dataset_provenance,
    seed_everything,
)
from nowcasting.models.survival_intensity_adapter import (  # noqa: E402
    DenseSurvivalIntensityAdapter,
    causal_pwv_state,
    radar_state_features,
)


HORIZONS = (("0h-1h", 0, 10), ("1h-2h", 10, 20))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Frozen-radar dense survival/intensity PWV adapter pilot"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pwv_root", required=True)
    parser.add_argument("--radar_checkpoint", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=39)
    parser.add_argument("--evaluation_lead_frames", type=int, default=20)
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--img_ch", type=int, default=2)
    parser.add_argument("--model_name", default="NowcastNet")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--lead_time_embed_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--adapter_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_train_samples", type=int, default=2048)
    parser.add_argument("--max_val_samples", type=int, default=512)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--max_samples_strategy", choices=["head", "uniform"], default="uniform")
    parser.add_argument("--intensity_scale", type=float, default=35.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--pwv_intensity_scale", type=float, default=80.0)
    parser.add_argument("--pwv_pixel_min", type=float, default=0.0)
    parser.add_argument("--pwv_pixel_max", type=float, default=255.0)
    parser.add_argument("--pwv_invert", action="store_true")
    parser.add_argument("--pwv_history_minutes", type=float, default=180.0)
    parser.add_argument("--pwv_anchor_minutes", type=float, default=30.0)
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--require_contiguous", action="store_true")
    parser.add_argument("--strict_pwv", action="store_true")
    parser.add_argument("--no_pwv_sequence", action="store_true", default=True)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--max_correction", type=float, default=12.0)
    parser.add_argument("--candidate_threshold", type=float, default=0.5)
    parser.add_argument("--candidate_radius", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_soft_csi", type=float, default=0.5)
    parser.add_argument("--lambda_far", type=float, default=0.1)
    parser.add_argument("--lambda_correction", type=float, default=0.002)
    parser.add_argument("--thresholds", default="10,20,30")
    parser.add_argument("--selection_thresholds", default="10,20")
    parser.add_argument("--soft_csi_temperature", type=float, default=1.5)
    parser.add_argument("--bootstrap_repetitions", type=int, default=1000)
    parser.add_argument("--minimum_csi_delta", type=float, default=0.003)
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Evaluate saved best adapters on the locked test split without retraining.",
    )
    return parser


def _parse_floats(text):
    return [float(item) for item in text.split(",") if item.strip()]


@torch.no_grad()
def build_cache(model, loader, args, name):
    model.eval()
    parts = defaultdict(list)
    cases = []
    sample_ids = []
    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device)
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        forecast = model(frames)[..., 0][:, : args.evaluation_lead_frames]
        observed = frames[:, : args.input_length, :, :, 0]
        target = batch["target_frames"].float().to(args.device)[
            :, : args.evaluation_lead_frames
        ]
        history = batch["pwv_history_frames"].float()
        parts["radar_state"].append(radar_state_features(observed).cpu().half())
        parts["radar_forecast"].append(forecast.cpu().half())
        parts["target"].append(target.cpu().half())
        parts["pwv_history"].append(history.half())
        cases.extend([str(item) for item in batch["case_name"]])
        sample_ids.extend([str(item) for item in batch["sample_id"]])
        if step % 20 == 0:
            print("{} cached batches {}".format(name, step), flush=True)
    cache = {key: torch.cat(values) for key, values in parts.items()}
    cache["cases"] = cases
    cache["sample_ids"] = sample_ids
    return cache


def attach_auxiliary(train_cache, val_cache, scale):
    climatology = train_cache["pwv_history"].float().mean(
        dim=(0, 1), keepdim=True
    )
    for cache in (train_cache, val_cache):
        aligned = causal_pwv_state(
            cache["pwv_history"].float(), climatology, scale
        )
        static = torch.zeros_like(aligned)
        static[:, 0] = aligned[:, 0]
        cache["pwv_aligned"] = aligned.half()
        cache["pwv_static"] = static.half()
        del cache["pwv_history"]
    return climatology


def attach_auxiliary_with_climatology(cache, climatology, scale):
    aligned = causal_pwv_state(
        cache["pwv_history"].float(), climatology.float(), scale
    )
    static = torch.zeros_like(aligned)
    static[:, 0] = aligned[:, 0]
    cache["pwv_aligned"] = aligned.half()
    cache["pwv_static"] = static.half()
    del cache["pwv_history"]
    return cache


def _candidate_from_cached(adapter, cache, indices, device):
    # The exact observed sequence is unnecessary after its four radar summaries
    # have been cached: channel 2 is the observed maximum used by the support.
    radar_state = cache["radar_state"][indices].float().to(device)
    forecast = cache["radar_forecast"][indices].float().to(device)
    support = torch.maximum(
        radar_state[:, 2:3], forecast.amax(dim=1, keepdim=True)
    )
    support = (support >= adapter.candidate_threshold).to(forecast.dtype)
    if adapter.candidate_radius > 0:
        kernel = 2 * adapter.candidate_radius + 1
        support = F.max_pool2d(
            support, kernel_size=kernel, stride=1, padding=adapter.candidate_radius
        )
    return radar_state, forecast, support


def adapter_forward_cached(adapter, cache, indices, auxiliary_key, device):
    radar_state, forecast, candidate = _candidate_from_cached(
        adapter, cache, indices, device
    )
    if auxiliary_key:
        auxiliary = cache[auxiliary_key][indices].float().to(device)
    else:
        auxiliary = forecast.new_zeros(
            forecast.shape[0], adapter.auxiliary_channels, *forecast.shape[-2:]
        )
    return adapter.forward_from_state(
        radar_state, forecast, auxiliary, candidate=candidate
    )


def soft_csi_loss(prediction, target, thresholds, temperature, lead_weights):
    losses = []
    for threshold in thresholds:
        probability = torch.sigmoid((prediction - threshold) / temperature)
        truth = (target >= threshold).to(prediction.dtype)
        hit = (probability * truth * lead_weights).sum()
        miss = ((1.0 - probability) * truth * lead_weights).sum()
        false_alarm = (probability * (1.0 - truth) * lead_weights).sum()
        losses.append(1.0 - hit / (hit + miss + false_alarm).clamp_min(1.0))
    return torch.stack(losses).mean()


def adapter_loss(result, target, args, selection_thresholds):
    lead_weights = target.new_ones((1, target.shape[1], 1, 1))
    lead_weights[:, 10:] = 2.0
    intensity_weights = (
        1.0
        + 2.0 * (target >= 10.0).to(target.dtype)
        + 3.0 * (target >= 20.0).to(target.dtype)
    )
    reconstruction = (
        F.smooth_l1_loss(result["prediction"], target, reduction="none")
        * lead_weights
        * intensity_weights
    ).mean()
    csi = soft_csi_loss(
        result["prediction"],
        target,
        selection_thresholds,
        args.soft_csi_temperature,
        lead_weights,
    )
    dry = (target < 1.0).to(target.dtype)
    false_alarm = (
        F.relu(result["prediction"] - 1.0) * dry * lead_weights
    ).sum() / (dry * lead_weights).sum().clamp_min(1.0)
    correction = result["correction"].abs().mean()
    total = (
        reconstruction
        + args.lambda_soft_csi * csi
        + args.lambda_far * false_alarm
        + args.lambda_correction * correction
    )
    return total, {
        "total": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "soft_csi": float(csi.detach()),
        "false_alarm": float(false_alarm.detach()),
        "correction_l1": float(correction.detach()),
    }


def _iter_indices(count, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size]


def counts_for_arrays(prediction, target, thresholds):
    result = {}
    for threshold in thresholds:
        pred = prediction >= threshold
        truth = target >= threshold
        result["{:g}".format(threshold)] = {
            "hit": int((pred & truth).sum()),
            "miss": int((~pred & truth).sum()),
            "false_alarm": int((pred & ~truth).sum()),
        }
    return result


def finalize_counts(counts):
    hit = counts["hit"]
    miss = counts["miss"]
    false_alarm = counts["false_alarm"]
    return {
        **counts,
        "csi": hit / max(hit + miss + false_alarm, 1),
        "pod": hit / max(hit + miss, 1),
        "far": false_alarm / max(hit + false_alarm, 1),
    }


@torch.no_grad()
def evaluate(adapter, cache, auxiliary_key, thresholds, args, donor_indices=None):
    adapter.eval()
    totals = {
        horizon: {
            "{:g}".format(threshold): {"hit": 0, "miss": 0, "false_alarm": 0}
            for threshold in thresholds
        }
        for horizon, _, _ in HORIZONS
    }
    by_case = defaultdict(
        lambda: {
            horizon: {
                "{:g}".format(threshold): {"hit": 0, "miss": 0, "false_alarm": 0}
                for threshold in thresholds
            }
            for horizon, _, _ in HORIZONS
        }
    )
    correction_sum = 0.0
    correction_count = 0
    for indices in _iter_indices(
        len(cache["cases"]), args.adapter_batch_size, False, args.seed
    ):
        working = cache
        if donor_indices is not None and auxiliary_key:
            working = dict(cache)
            working[auxiliary_key] = cache[auxiliary_key][donor_indices]
        result = adapter_forward_cached(
            adapter, working, indices, auxiliary_key, args.device
        )
        target = cache["target"][indices].float().to(args.device)
        correction_sum += float(result["correction"].abs().sum())
        correction_count += result["correction"].numel()
        for horizon, start, end in HORIZONS:
            batch_counts = counts_for_arrays(
                result["prediction"][:, start:end],
                target[:, start:end],
                thresholds,
            )
            for key, values in batch_counts.items():
                for field, value in values.items():
                    totals[horizon][key][field] += value
            for local, sample_index in enumerate(indices.tolist()):
                sample_counts = counts_for_arrays(
                    result["prediction"][local, start:end],
                    target[local, start:end],
                    thresholds,
                )
                case = cache["cases"][sample_index]
                for key, values in sample_counts.items():
                    for field, value in values.items():
                        by_case[case][horizon][key][field] += value
    return {
        "horizons": {
            horizon: {
                key: finalize_counts(values) for key, values in threshold_values.items()
            }
            for horizon, threshold_values in totals.items()
        },
        "by_case": dict(by_case),
        "mean_abs_correction": correction_sum / max(correction_count, 1),
    }


def selection_score(evaluation, thresholds):
    return sum(
        evaluation["horizons"]["1h-2h"]["{:g}".format(threshold)]["csi"]
        for threshold in thresholds
    )


def train_adapter(cache, val_cache, auxiliary_key, initial_state, args, thresholds):
    adapter = DenseSurvivalIntensityAdapter(
        hidden_channels=args.hidden_channels,
        max_correction_mm_per_h=args.max_correction,
        candidate_threshold_mm_per_h=args.candidate_threshold,
        candidate_radius=args.candidate_radius,
    ).to(args.device)
    adapter.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        sums = defaultdict(float)
        batches = 0
        for indices in _iter_indices(
            len(cache["cases"]),
            args.adapter_batch_size,
            True,
            args.seed + epoch,
        ):
            target = cache["target"][indices].float().to(args.device)
            result = adapter_forward_cached(
                adapter, cache, indices, auxiliary_key, args.device
            )
            loss, parts = adapter_loss(result, target, args, thresholds)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            for key, value in parts.items():
                sums[key] += value
            batches += 1
        validation = evaluate(
            adapter, val_cache, auxiliary_key, thresholds, args
        )
        score = selection_score(validation, thresholds)
        row = {
            "epoch": epoch,
            **{key: value / max(batches, 1) for key, value in sums.items()},
            "val_csi10_plus_csi20_second_hour": score,
        }
        history.append(row)
        print(
            "epoch {:02d} {} score {:.6f} loss {:.6f}".format(
                epoch, auxiliary_key or "radar_only", score, row["total"]
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
    adapter.load_state_dict(best_state)
    return adapter, history, best_score


def cross_event_indices(cases):
    count = len(cases)
    result = []
    for index, case in enumerate(cases):
        donor = (index + max(1, count // 2)) % count
        searched = 0
        while cases[donor] == case and searched < count:
            donor = (donor + 1) % count
            searched += 1
        if cases[donor] == case:
            raise ValueError("Cross-event control requires at least two cases.")
        result.append(donor)
    return torch.tensor(result, dtype=torch.long)


def bootstrap_delta(left, right, horizon, key, repetitions, seed):
    cases = sorted(set(left["by_case"]) & set(right["by_case"]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        deltas = []
        aggregate_left = {"hit": 0, "miss": 0, "false_alarm": 0}
        aggregate_right = {"hit": 0, "miss": 0, "false_alarm": 0}
        for case in sampled:
            for field in aggregate_left:
                aggregate_left[field] += left["by_case"][case][horizon][key][field]
                aggregate_right[field] += right["by_case"][case][horizon][key][field]
        deltas.append(
            finalize_counts(aggregate_left)["csi"]
            - finalize_counts(aggregate_right)["csi"]
        )
        values.extend(deltas)
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
        "iterations": len(values),
        "clusters": len(cases),
    }


def compare(left, right, thresholds, args):
    result = {}
    for horizon, _, _ in HORIZONS:
        result[horizon] = {}
        for threshold in thresholds:
            key = "{:g}".format(threshold)
            left_metrics = left["horizons"][horizon][key]
            right_metrics = right["horizons"][horizon][key]
            result[horizon][key] = {
                "csi_delta": left_metrics["csi"] - right_metrics["csi"],
                "pod_delta": left_metrics["pod"] - right_metrics["pod"],
                "far_delta": left_metrics["far"] - right_metrics["far"],
                "case_bootstrap_csi_delta": bootstrap_delta(
                    left,
                    right,
                    horizon,
                    key,
                    args.bootstrap_repetitions,
                    args.seed,
                ),
            }
    return result


def promotion_gate(comparisons, minimum_delta):
    tasks = {}
    passing = 0
    for key in ("10", "20"):
        against_radar = comparisons["pwv_aligned_minus_radar_adapter"]["1h-2h"][key]
        against_cross = comparisons["pwv_aligned_minus_pwv_cross_event"]["1h-2h"][key]
        against_static = comparisons["pwv_aligned_minus_pwv_static"]["1h-2h"][key]
        passed = (
            against_radar["csi_delta"] >= minimum_delta
            and against_cross["csi_delta"] >= minimum_delta
            and against_static["csi_delta"] >= minimum_delta
            and against_radar["far_delta"] <= 0.02
        )
        tasks[key] = {
            "pass": passed,
            "against_radar_adapter": against_radar,
            "against_cross_event": against_cross,
            "against_static": against_static,
        }
        passing += int(passed)
    return {
        "minimum_csi_delta": minimum_delta,
        "required_tasks": 2,
        "passing_tasks": passing,
        "provisional_gate": "pass" if passing == 2 else "fail",
        "tasks": tasks,
    }


def load_adapter(path, args):
    adapter = DenseSurvivalIntensityAdapter(
        hidden_channels=args.hidden_channels,
        max_correction_mm_per_h=args.max_correction,
        candidate_threshold_mm_per_h=args.candidate_threshold,
        candidate_radius=args.candidate_radius,
    ).to(args.device)
    payload = torch.load(path, map_location=args.device)
    adapter.load_state_dict(payload["model"] if "model" in payload else payload)
    return adapter


def evaluate_locked_split(
    radar_adapter,
    pwv_adapter,
    identity_adapter,
    cache,
    thresholds,
    args,
):
    variants = {
        "radar_backbone": evaluate(
            identity_adapter, cache, "", thresholds, args
        ),
        "radar_adapter": evaluate(
            radar_adapter, cache, "", thresholds, args
        ),
        "pwv_aligned": evaluate(
            pwv_adapter, cache, "pwv_aligned", thresholds, args
        ),
        "pwv_static": evaluate(
            pwv_adapter, cache, "pwv_static", thresholds, args
        ),
        "pwv_cross_event": evaluate(
            pwv_adapter,
            cache,
            "pwv_aligned",
            thresholds,
            args,
            donor_indices=cross_event_indices(cache["cases"]),
        ),
    }
    comparisons = {
        "radar_adapter_minus_radar_backbone": compare(
            variants["radar_adapter"],
            variants["radar_backbone"],
            thresholds,
            args,
        ),
        "pwv_aligned_minus_radar_backbone": compare(
            variants["pwv_aligned"],
            variants["radar_backbone"],
            thresholds,
            args,
        ),
        "pwv_aligned_minus_radar_adapter": compare(
            variants["pwv_aligned"], variants["radar_adapter"], thresholds, args
        ),
        "pwv_aligned_minus_pwv_static": compare(
            variants["pwv_aligned"], variants["pwv_static"], thresholds, args
        ),
        "pwv_aligned_minus_pwv_cross_event": compare(
            variants["pwv_aligned"],
            variants["pwv_cross_event"],
            thresholds,
            args,
        ),
    }
    return variants, comparisons


def run_test_only(args, output, thresholds):
    cache_path = output / "frozen_radar_cache.pt"
    if not cache_path.exists():
        raise FileNotFoundError("training cache is required: {}".format(cache_path))
    payload = torch.load(cache_path, map_location="cpu")
    climatology = payload["climatology"]
    test_cache_path = output / "frozen_radar_test_cache.pt"
    test_loader = make_png_dataloader(
        args, "test", args.max_test_samples, shuffle=False, drop_last=False
    )
    save_dataset_provenance(
        {"test": test_loader}, output / "test_data_manifest.json"
    )
    if args.reuse_cache and test_cache_path.exists():
        test_cache = torch.load(test_cache_path, map_location="cpu")
        print("reused {}".format(test_cache_path), flush=True)
    else:
        radar = build_generator(args)
        load_generator_weights(radar, args.radar_checkpoint, args.device)
        for parameter in radar.parameters():
            parameter.requires_grad_(False)
        test_cache = build_cache(radar, test_loader, args, "test")
        del radar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        attach_auxiliary_with_climatology(
            test_cache, climatology, args.pwv_intensity_scale
        )
        safe_torch_save(test_cache, test_cache_path)
    radar_adapter = load_adapter(output / "radar_adapter.ckpt", args)
    pwv_adapter = load_adapter(output / "pwv_adapter.ckpt", args)
    identity_adapter = DenseSurvivalIntensityAdapter(
        hidden_channels=args.hidden_channels,
        max_correction_mm_per_h=args.max_correction,
        candidate_threshold_mm_per_h=args.candidate_threshold,
        candidate_radius=args.candidate_radius,
    ).to(args.device)
    variants, comparisons = evaluate_locked_split(
        radar_adapter,
        pwv_adapter,
        identity_adapter,
        test_cache,
        thresholds,
        args,
    )
    report = {
        "protocol": "pwv_survival_intensity_adapter_pilot",
        "role": "locked_held_out_test",
        "checkpoint_selection": "fixed from validation; no test-set tuning",
        "samples": {
            "test": len(test_cache["cases"]),
            "test_case_clusters": len(set(test_cache["cases"])),
        },
        "variants": variants,
        "comparisons": comparisons,
        "promotion_gate": promotion_gate(comparisons, args.minimum_csi_delta),
        "warning": (
            "Do not modify the model or thresholds after inspecting this result; "
            "a successor requires repeated seeds or a newly held-out split."
        ),
    }
    path = output / "test_metrics.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_json_numbers(report), handle, indent=2)
    print(json.dumps(report["promotion_gate"], indent=2), flush=True)
    print("wrote {}".format(path), flush=True)


def main():
    args = add_model_runtime_args(build_parser().parse_args())
    if args.evaluation_lead_frames != 20:
        raise ValueError("This locked pilot evaluates exactly 20 frames (0--2 h).")
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = _parse_floats(args.thresholds)
    if args.test_only:
        run_test_only(args, output, thresholds)
        return
    cache_path = output / "frozen_radar_cache.pt"
    train_loader = make_png_dataloader(
        args, "train", args.max_train_samples, shuffle=False, drop_last=False
    )
    val_loader = make_png_dataloader(
        args, "val", args.max_val_samples, shuffle=False, drop_last=False
    )
    save_dataset_provenance(
        {"train": train_loader, "val": val_loader}, output / "data_manifest.json"
    )
    if args.reuse_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        train_cache, val_cache = payload["train"], payload["val"]
        climatology = payload["climatology"]
        print("reused {}".format(cache_path), flush=True)
    else:
        radar = build_generator(args)
        load_generator_weights(radar, args.radar_checkpoint, args.device)
        for parameter in radar.parameters():
            parameter.requires_grad_(False)
        train_cache = build_cache(radar, train_loader, args, "train")
        val_cache = build_cache(radar, val_loader, args, "val")
        del radar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        climatology = attach_auxiliary(
            train_cache, val_cache, args.pwv_intensity_scale
        )
        safe_torch_save(
            {
                "train": train_cache,
                "val": val_cache,
                "climatology": climatology,
            },
            cache_path,
        )
    selection_thresholds = _parse_floats(args.selection_thresholds)
    template = DenseSurvivalIntensityAdapter(
        hidden_channels=args.hidden_channels,
        max_correction_mm_per_h=args.max_correction,
        candidate_threshold_mm_per_h=args.candidate_threshold,
        candidate_radius=args.candidate_radius,
    )
    initial_state = {
        key: value.detach().clone() for key, value in template.state_dict().items()
    }
    radar_adapter, radar_history, radar_score = train_adapter(
        train_cache,
        val_cache,
        "",
        initial_state,
        args,
        selection_thresholds,
    )
    pwv_adapter, pwv_history, pwv_score = train_adapter(
        train_cache,
        val_cache,
        "pwv_aligned",
        initial_state,
        args,
        selection_thresholds,
    )
    safe_torch_save(
        {"model": radar_adapter.state_dict(), "history": radar_history},
        output / "radar_adapter.ckpt",
    )
    safe_torch_save(
        {"model": pwv_adapter.state_dict(), "history": pwv_history},
        output / "pwv_adapter.ckpt",
    )
    radar_backbone_identity = DenseSurvivalIntensityAdapter(
        hidden_channels=args.hidden_channels,
        max_correction_mm_per_h=args.max_correction,
        candidate_threshold_mm_per_h=args.candidate_threshold,
        candidate_radius=args.candidate_radius,
    ).to(args.device)
    radar_backbone_identity.load_state_dict(initial_state)
    variants, comparisons = evaluate_locked_split(
        radar_adapter,
        pwv_adapter,
        radar_backbone_identity,
        val_cache,
        thresholds,
        args,
    )
    report = {
        "protocol": "pwv_survival_intensity_adapter_pilot",
        "role": "development_pilot",
        "samples": {
            "train": len(train_cache["cases"]),
            "validation": len(val_cache["cases"]),
            "validation_case_clusters": len(set(val_cache["cases"])),
        },
        "checkpoint_selection": {
            "objective": "validation CSI10 + CSI20 at 1h-2h",
            "radar_adapter_best_score": radar_score,
            "pwv_adapter_best_score": pwv_score,
        },
        "variants": variants,
        "comparisons": comparisons,
        "promotion_gate": promotion_gate(comparisons, args.minimum_csi_delta),
        "warnings": [
            "PWV uses causal 30-minute anchors from the available processed product.",
            "A pass is provisional until repeated seeds and held-out test events agree.",
        ],
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(sanitize_json_numbers(report), handle, indent=2)
    with open(output / "train_history.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"radar_adapter": radar_history, "pwv_adapter": pwv_history},
            handle,
            indent=2,
        )
    print(json.dumps(report["promotion_gate"], indent=2), flush=True)
    print("wrote {}".format(output / "metrics.json"), flush=True)


if __name__ == "__main__":
    main()
