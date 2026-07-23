"""Test causal multi-hour PWV preconditioning after frozen radar history.

This successor to ``pwv_conditional_probe.py`` treats the interpolated
six-minute PWV images as a low-frequency environmental state.  Its primary PWV
representation uses only 30-minute anchors at or before the radar issue time,
so an interpolated image cannot indirectly consume a future native endpoint.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.pwv_conditional_probe import (
    HORIZONS,
    ConditionalEventProbe,
    _batch_indices,
    _curve_from_histogram,
    _event_targets,
    _frozen_radar_latent,
    _histogram,
    _pwv_features,
    build_parser as build_base_parser,
    cross_event_indices,
    event_metrics,
    optional_delta,
    paired_day_bootstrap,
    split_fit_calibration,
    train_probe,
)
from nowcasting.experiments.common import (
    add_model_runtime_args,
    build_generator,
    load_generator_weights,
    load_model_state,
    make_png_dataloader,
    safe_torch_save,
    sanitize_json_numbers,
    save_dataset_provenance,
    seed_everything,
)


PRIMARY_STRATUM = "weak_echo_nondecreasing"


def build_parser():
    parser = build_base_parser()
    parser.description = (
        "Causal multi-hour PWV preconditioning probe after frozen radar"
    )
    parser.add_argument("--pwv_history_minutes", type=float, default=180.0)
    parser.add_argument("--pwv_anchor_minutes", type=float, default=30.0)
    return parser


def _pool_series(series, latent_size, mode):
    batch, steps, height, width = series.shape
    flat = series.reshape(batch * steps, 1, height, width)
    if mode == "max":
        pooled = F.adaptive_max_pool2d(flat, latent_size)
    elif mode == "mean":
        pooled = F.adaptive_avg_pool2d(flat, latent_size)
    else:
        raise ValueError("Unknown pooling mode: {}".format(mode))
    return pooled.reshape(batch, steps, latent_size[0], latent_size[1])


@torch.no_grad()
def build_preconditioning_cache(model, loader, args, thresholds):
    model.eval()
    parts = defaultdict(list)
    cases = []
    sample_ids = []
    history_ranges = []
    for step, batch in enumerate(loader, 1):
        frames = batch["radar_frames"].float().to(args.device)
        target = batch["target_frames"].float().to(args.device)
        pwv = batch["pwv_frames"].float().to(args.device)
        history = batch["pwv_history_frames"].float().to(args.device)
        radar = _frozen_radar_latent(model, frames, args)
        latent_size = radar.shape[-2:]
        observed_radar = frames[..., 0][:, : args.input_length]

        parts["radar"].append(radar.cpu().half())
        parts["pwv_short"].append(
            _pwv_features(pwv, args, latent_size).cpu().half()
        )
        parts["pwv_history"].append(
            _pool_series(history, latent_size, "mean").cpu().half()
        )
        parts["observed_radar_tiles"].append(
            _pool_series(observed_radar, latent_size, "max").cpu().half()
        )
        parts["target"].append(
            _event_targets(target, thresholds, latent_size).cpu()
        )
        cases.extend(list(batch["case_name"]))
        sample_ids.extend(list(batch["sample_id"]))
        history_ranges.extend(
            zip(
                list(batch["pwv_history_start_file"]),
                list(batch["pwv_history_end_file"]),
            )
        )
        if step % 20 == 0:
            print("cached batches {}".format(step), flush=True)
    cache = {key: torch.cat(value) for key, value in parts.items()}
    cache.update(
        {
            "cases": cases,
            "sample_ids": sample_ids,
            "pwv_history_ranges": [list(item) for item in history_ranges],
        }
    )
    return cache


def causal_pwv_features(history, climatology, scale):
    """Make six matched channels from native-cadence causal PWV anchors."""
    history = history.float()
    last = history[:, -1:]
    mean = history.mean(dim=1, keepdim=True)
    std = history.std(dim=1, keepdim=True, unbiased=False)
    tendency = last - history[:, :1]
    anomaly = last - climatology
    dx = F.pad(last[..., 1:] - last[..., :-1], (0, 1, 0, 0))
    dy = F.pad(last[..., 1:, :] - last[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx.square() + dy.square() + 1e-8)
    return (
        torch.cat([last, mean, std, tendency, anomaly, gradient], dim=1)
        / max(float(scale), 1e-6)
    ).clamp(-2.0, 2.0)


def attach_long_features(caches, fit_indices, args):
    fit_history = caches["train"]["pwv_history"][fit_indices].float()
    climatology = fit_history.mean(dim=(0, 1), keepdim=True)
    for cache in caches.values():
        features = causal_pwv_features(
            cache["pwv_history"],
            climatology,
            args.pwv_intensity_scale,
        )
        reversed_tendency = features.clone()
        reversed_tendency[:, 3:4] = -reversed_tendency[:, 3:4]
        cache["pwv_long"] = features.half()
        cache["pwv_long_tendency_reversed"] = reversed_tendency.half()
    return climatology


def build_strata(cache, thresholds):
    observed = cache["observed_radar_tiles"].float()
    target = cache["target"].bool()
    last = observed[:, -1]
    early = observed[:, : min(3, observed.shape[1])].mean(dim=1)
    history_max = observed.max(dim=1).values
    masks = {
        "all": torch.ones_like(target, dtype=torch.bool),
        PRIMARY_STRATUM: torch.zeros_like(target, dtype=torch.bool),
        "radar_quiet": torch.zeros_like(target, dtype=torch.bool),
    }
    for threshold_index, threshold in enumerate(thresholds):
        weak_echo = (
            (last >= 1.0)
            & (last < threshold)
            & (last >= early)
        )
        quiet = history_max < 1.0
        masks[PRIMARY_STRATUM][:, :, threshold_index] = weak_echo[:, None]
        masks["radar_quiet"][:, :, threshold_index] = quiet[:, None]
    return masks


@torch.no_grad()
def predict(
    model,
    cache,
    use_pwv,
    args,
    feature_key,
    control="real",
):
    model.to(args.device).eval()
    outputs = []
    count = len(cache["radar"])
    donor = cross_event_indices(cache["cases"])
    for indices in _batch_indices(
        torch.arange(count), args.probe_batch_size, False, args.seed
    ):
        radar = cache["radar"][indices].float().to(args.device)
        if control == "cross_event":
            auxiliary = cache[feature_key][donor[indices]].float().to(args.device)
        else:
            auxiliary = cache[feature_key][indices].float().to(args.device)
        if control == "spatial_shift":
            auxiliary = torch.roll(
                auxiliary,
                shifts=(
                    max(1, auxiliary.shape[-2] // 2),
                    max(1, auxiliary.shape[-1] // 2),
                ),
                dims=(-2, -1),
            )
        outputs.append(
            torch.sigmoid(
                model(radar, auxiliary, use_pwv=use_pwv)
            ).cpu()
        )
    return torch.cat(outputs)


def calibrate(
    model,
    cache,
    indices,
    use_pwv,
    feature_key,
    thresholds,
    args,
):
    subset = {
        key: (
            value[indices]
            if torch.is_tensor(value)
            else [value[int(index)] for index in indices]
        )
        for key, value in cache.items()
    }
    probabilities = predict(
        model, subset, use_pwv, args, feature_key=feature_key
    )
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
            best = int(torch.argmax(csi))
            selected[horizon]["{:g}".format(threshold)] = {
                "probability_threshold": best / args.histogram_bins,
                "calibration_csi": float(csi[best]),
            }
    return selected


def evaluate(
    model,
    cache,
    use_pwv,
    feature_key,
    control,
    cutoffs,
    thresholds,
    args,
):
    probabilities = predict(
        model,
        cache,
        use_pwv,
        args,
        feature_key=feature_key,
        control=control,
    )
    target = cache["target"].bool()
    strata = build_strata(cache, thresholds)
    summary = {"control": control, "strata": {}, "eventwise": {}}
    for stratum_name, stratum_mask in strata.items():
        summary["strata"][stratum_name] = {}
        rows = [
            {
                "sample_id": sample_id,
                "case_name": cache["cases"][sample],
                "events": {},
            }
            for sample, sample_id in enumerate(cache["sample_ids"])
        ]
        for horizon, start, end in HORIZONS:
            summary["strata"][stratum_name][horizon] = {}
            for threshold_index, threshold in enumerate(thresholds):
                key = "{:g}".format(threshold)
                probability = probabilities[:, start:end, threshold_index]
                truth = target[:, start:end, threshold_index]
                mask = stratum_mask[:, start:end, threshold_index]
                cutoff = cutoffs[horizon][key]["probability_threshold"]
                metrics = event_metrics(
                    probability[mask],
                    truth[mask],
                    cutoff,
                )
                positive, negative = _histogram(
                    probability[mask],
                    truth[mask],
                    args.histogram_bins,
                )
                _, _, _, average_precision = _curve_from_histogram(
                    positive, negative
                )
                metrics.update(
                    {
                        "average_precision": float(average_precision),
                        "probability_threshold": cutoff,
                        "candidate_cells": int(mask.sum()),
                        "positive_cells": int((truth & mask).sum()),
                    }
                )
                summary["strata"][stratum_name][horizon][key] = metrics
                event_key = "{}::{}".format(horizon, key)
                for sample in range(len(rows)):
                    rows[sample]["events"][event_key] = event_metrics(
                        probability[sample][mask[sample]],
                        truth[sample][mask[sample]],
                        cutoff,
                    )
        summary["eventwise"][stratum_name] = rows
    return summary


def compare_variants(variants, thresholds, args):
    references = (
        "radar_only",
        "pwv_short_interpolated",
        "pwv_long_spatial_shift",
        "pwv_long_cross_event",
        "pwv_long_tendency_reversed",
    )
    comparisons = {}
    for reference in references:
        by_stratum = {}
        for stratum in ("all", PRIMARY_STRATUM, "radar_quiet"):
            by_stratum[stratum] = {}
            for horizon, _, _ in HORIZONS:
                by_stratum[stratum][horizon] = {}
                for threshold in thresholds:
                    key = "{:g}".format(threshold)
                    event_key = "{}::{}".format(horizon, key)
                    left = variants["pwv_long_causal"]["strata"][stratum][horizon][key]
                    right = variants[reference]["strata"][stratum][horizon][key]
                    by_stratum[stratum][horizon][key] = {
                        "csi_delta": optional_delta(left["csi"], right["csi"]),
                        "average_precision_delta": optional_delta(
                            left["average_precision"],
                            right["average_precision"],
                        ),
                        "day_cluster_bootstrap_csi_delta": paired_day_bootstrap(
                            variants["pwv_long_causal"]["eventwise"][stratum],
                            variants[reference]["eventwise"][stratum],
                            event_key,
                            args.bootstrap_repetitions,
                            args.seed,
                        ),
                    }
        comparisons["pwv_long_causal_minus_" + reference] = by_stratum
    return comparisons


def promotion_gate(comparisons, thresholds, minimum_delta):
    required_references = (
        "radar_only",
        "pwv_short_interpolated",
        "pwv_long_spatial_shift",
        "pwv_long_cross_event",
    )
    tasks = {}
    passing = 0
    for horizon, _, _ in HORIZONS:
        for threshold in thresholds:
            key = "{:g}".format(threshold)
            name = "{}::{}".format(horizon, key)
            reference_results = {}
            task_pass = True
            for reference in required_references:
                item = comparisons[
                    "pwv_long_causal_minus_" + reference
                ][PRIMARY_STRATUM][horizon][key]
                passed = (
                    item["csi_delta"] is not None
                    and item["csi_delta"] >= minimum_delta
                    and item["average_precision_delta"] is not None
                    and item["average_precision_delta"] > 0
                )
                reference_results[reference] = passed
                task_pass = task_pass and passed
            tasks[name] = {
                "pass": task_pass,
                "references": reference_results,
            }
            passing += int(task_pass)

    all_window_safety = True
    for horizon, _, _ in HORIZONS:
        for threshold in thresholds:
            key = "{:g}".format(threshold)
            delta = comparisons[
                "pwv_long_causal_minus_radar_only"
            ]["all"][horizon][key]["csi_delta"]
            if delta is not None and delta < -minimum_delta:
                all_window_safety = False
    return {
        "primary_stratum": PRIMARY_STRATUM,
        "minimum_csi_delta": minimum_delta,
        "required_passing_tasks": 3,
        "passing_tasks": passing,
        "all_window_safety_pass": all_window_safety,
        "point_estimate_pass": passing >= 3 and all_window_safety,
        "tasks": tasks,
    }


def main():
    args = add_model_runtime_args(build_parser().parse_args())
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
    cache_path = output / "preconditioning_feature_cache.pt"
    if args.reuse_cache and cache_path.exists():
        caches = torch.load(cache_path, map_location="cpu")
    else:
        radar = build_generator(args)
        load_generator_weights(
            radar, args.radar_checkpoint, args.device, strict=True
        )
        radar.requires_grad_(False)
        caches = {
            "train": build_preconditioning_cache(
                radar, train_loader, args, thresholds
            ),
            "val": build_preconditioning_cache(
                radar, val_loader, args, thresholds
            ),
        }
        safe_torch_save(caches, cache_path)
        del radar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fit_indices, calibration_indices = split_fit_calibration(
        caches["train"]["cases"], args.calibration_day_fraction
    )
    climatology = attach_long_features(caches, fit_indices, args)
    feature_keys = {
        "radar": "pwv_long",
        "short": "pwv_short",
        "long": "pwv_long",
    }
    probe_kwargs = {
        "radar_channels": caches["train"]["radar"].shape[1],
        "pwv_channels": caches["train"]["pwv_long"].shape[1],
        "hidden_channels": args.hidden_channels,
        "lead_count": args.evo_ic,
        "threshold_count": len(thresholds),
    }
    probes = {}
    for name in ("radar", "short", "long"):
        seed_everything(args.seed)
        probes[name] = ConditionalEventProbe(**probe_kwargs)
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in probes.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise AssertionError("All probes must have identical parameter counts.")

    probes["radar"] = train_probe(
        probes["radar"],
        caches["train"],
        fit_indices,
        False,
        args,
        "radar",
        feature_key=feature_keys["radar"],
    )
    probes["short"] = train_probe(
        probes["short"],
        caches["train"],
        fit_indices,
        True,
        args,
        "short",
        feature_key=feature_keys["short"],
    )
    probes["long"] = train_probe(
        probes["long"],
        caches["train"],
        fit_indices,
        True,
        args,
        "long",
        feature_key=feature_keys["long"],
    )
    cutoffs = {
        name: calibrate(
            probes[name],
            caches["train"],
            calibration_indices,
            name != "radar",
            feature_keys[name],
            thresholds,
            args,
        )
        for name in probes
    }

    specifications = {
        "radar_only": ("radar", False, "pwv_long", "learned_constant"),
        "pwv_short_interpolated": ("short", True, "pwv_short", "real"),
        "pwv_long_causal": ("long", True, "pwv_long", "real"),
        "pwv_long_spatial_shift": ("long", True, "pwv_long", "spatial_shift"),
        "pwv_long_cross_event": ("long", True, "pwv_long", "cross_event"),
        "pwv_long_tendency_reversed": (
            "long",
            True,
            "pwv_long_tendency_reversed",
            "tendency_reversed",
        ),
    }
    variants = {}
    for variant, (probe_name, use_pwv, feature_key, control) in specifications.items():
        variants[variant] = evaluate(
            probes[probe_name],
            caches["val"],
            use_pwv,
            feature_key,
            control,
            cutoffs[probe_name],
            thresholds,
            args,
        )
    comparisons = compare_variants(variants, thresholds, args)
    gate = promotion_gate(
        comparisons, thresholds, args.minimum_csi_delta
    )
    for variant in variants.values():
        variant.pop("eventwise")

    summary = sanitize_json_numbers(
        {
            "protocol": "pwv_causal_preconditioning_probe",
            "pwv_history_minutes": args.pwv_history_minutes,
            "pwv_anchor_minutes": args.pwv_anchor_minutes,
            "causality": (
                "PWV anchors are aligned to native cadence and are never later "
                "than the final observed radar timestamp."
            ),
            "train_samples": len(caches["train"]["radar"]),
            "validation_samples": len(caches["val"]["radar"]),
            "fit_samples": len(fit_indices),
            "calibration_samples": len(calibration_indices),
            "probe_parameters": parameter_counts,
            "frozen_radar_parameters": sum(
                tensor.numel()
                for tensor in load_model_state(
                    args.radar_checkpoint, "cpu"
                ).values()
                if torch.is_tensor(tensor)
            ),
            "climatology": {
                "source": "fit days only",
                "mean_mm": float(climatology.mean()),
                "spatial_std_mm": float(climatology.std()),
            },
            "strata": {
                "all": "all validation tiles",
                PRIMARY_STRATUM: (
                    "last observed radar tile is 1 mm/h to below the task "
                    "threshold and is not lower than the mean of the first "
                    "three observed frames"
                ),
                "radar_quiet": (
                    "observed radar-history tile maximum is below 1 mm/h; "
                    "diagnostic only"
                ),
            },
            "calibration_thresholds": cutoffs,
            "variants": variants,
            "paired_comparisons": comparisons,
            "gate": gate,
            "interpretation": (
                "A pass supports multi-hour PWV preconditioning for a "
                "restricted adapter. It is not final dense-forecast evidence."
            ),
        }
    )
    (output / "preconditioning_probe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for name, model in probes.items():
        safe_torch_save(
            model.state_dict(),
            output / "{}_probe.ckpt".format(name),
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
