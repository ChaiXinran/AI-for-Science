"""Train the final fair static-climatology versus dynamic-PWV probe.

Both probes receive the exact same fit-day static PWV climatology channel.  The
dynamic probe additionally receives event-level scalar moisture and
event-specific spatial residual channels.  Radar features, architecture,
parameter count, initialization, optimization budget, targets, and data splits
are otherwise identical.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.pwv_conditional_probe import (  # noqa: E402
    HORIZONS,
    ConditionalEventProbe,
    optional_delta,
    paired_day_bootstrap,
    split_fit_calibration,
    train_probe,
)
from diagnostics.pwv_preconditioning_probe import (  # noqa: E402
    PRIMARY_STRATUM,
    calibrate,
    evaluate,
)
from nowcasting.experiments.common import (  # noqa: E402
    load_model_state,
    safe_torch_save,
    sanitize_json_numbers,
    seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Matched static-climatology versus dynamic-PWV residual probe"
    )
    parser.add_argument("--seed_root", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--thresholds", default="10,20")
    parser.add_argument("--histogram_bins", type=int, default=1000)
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--calibration_day_fraction", type=float, default=0.2)
    parser.add_argument("--pwv_intensity_scale", type=float, default=80.0)
    parser.add_argument("--minimum_csi_delta", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def fair_static_dynamic_features(history, climatology, scale):
    """Create matched six-channel static and dynamic representations.

    Channels:
      0. shared static climatology;
      1. latest event scalar moisture departure;
      2. mean event scalar departure over three hours;
      3. three-hour event scalar tendency;
      4. latest zero-mean event spatial anomaly;
      5. three-hour event spatial-anomaly tendency.
    """
    history = history.float()
    climatology = climatology.float()
    valid = (climatology > 1e-6).to(history.dtype)
    valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    residual = (history - climatology) * valid
    scalar = residual.sum(dim=(-2, -1), keepdim=True) / valid_count
    anomaly = (residual - scalar * valid) * valid
    batch = history.shape[0]
    static_channel = climatology.expand(batch, -1, -1, -1)
    scalar_last = scalar[:, -1] * valid[:, 0]
    scalar_mean = scalar.mean(dim=1) * valid[:, 0]
    scalar_tendency = (scalar[:, -1] - scalar[:, 0]) * valid[:, 0]
    anomaly_last = anomaly[:, -1]
    anomaly_tendency = anomaly[:, -1] - anomaly[:, 0]
    dynamic = torch.stack(
        [
            static_channel[:, 0],
            scalar_last,
            scalar_mean,
            scalar_tendency,
            anomaly_last,
            anomaly_tendency,
        ],
        dim=1,
    ) / max(float(scale), 1e-6)
    static = torch.zeros_like(dynamic)
    static[:, 0] = dynamic[:, 0]
    diagnostics = {
        "valid_fraction": float(valid.mean()),
        "mean_abs_scalar_latest_mm": float(scalar[:, -1].abs().mean()),
        "mean_abs_scalar_tendency_mm": float(
            (scalar[:, -1] - scalar[:, 0]).abs().mean()
        ),
        "mean_spatial_anomaly_std_mm": float(
            anomaly_last.std(dim=(-2, -1), unbiased=False).mean()
        ),
        "mean_spatial_anomaly_tendency_std_mm": float(
            anomaly_tendency.std(dim=(-2, -1), unbiased=False).mean()
        ),
    }
    return static.clamp(-2.0, 2.0), dynamic.clamp(-2.0, 2.0), diagnostics


def attach_features(caches, fit_indices, scale):
    climatology = caches["train"]["pwv_history"][fit_indices].float().mean(
        dim=(0, 1), keepdim=True
    )
    diagnostics = {}
    for split, cache in caches.items():
        static, dynamic, split_diagnostics = fair_static_dynamic_features(
            cache["pwv_history"], climatology, scale
        )
        cache["pwv_static_fair"] = static.half()
        cache["pwv_dynamic_residual"] = dynamic.half()
        shifted = dynamic.clone()
        shifted[:, 4:6] = torch.roll(
            shifted[:, 4:6],
            shifts=(
                max(1, shifted.shape[-2] // 2),
                max(1, shifted.shape[-1] // 2),
            ),
            dims=(-2, -1),
        )
        cache["pwv_dynamic_spatial_shift"] = shifted.half()
        reversed_tendency = dynamic.clone()
        reversed_tendency[:, 3:4] = -reversed_tendency[:, 3:4]
        reversed_tendency[:, 5:6] = -reversed_tendency[:, 5:6]
        cache["pwv_dynamic_tendency_reversed"] = reversed_tendency.half()
        diagnostics[split] = split_diagnostics
    return climatology, diagnostics


def compare(left_name, right_name, variants, thresholds, args):
    result = {}
    for stratum in ("all", PRIMARY_STRATUM, "radar_quiet"):
        result[stratum] = {}
        for horizon, _, _ in HORIZONS:
            result[stratum][horizon] = {}
            for threshold in thresholds:
                key = "{:g}".format(threshold)
                event_key = "{}::{}".format(horizon, key)
                left = variants[left_name]["strata"][stratum][horizon][key]
                right = variants[right_name]["strata"][stratum][horizon][key]
                result[stratum][horizon][key] = {
                    "csi_delta": optional_delta(left["csi"], right["csi"]),
                    "average_precision_delta": optional_delta(
                        left["average_precision"],
                        right["average_precision"],
                    ),
                    "day_cluster_bootstrap_csi_delta": paired_day_bootstrap(
                        variants[left_name]["eventwise"][stratum],
                        variants[right_name]["eventwise"][stratum],
                        event_key,
                        args.bootstrap_repetitions,
                        args.seed,
                    ),
                }
    return result


def promotion_gate(comparisons, thresholds, minimum_delta):
    references = (
        "static_climatology_trained",
        "dynamic_cross_event",
        "dynamic_spatial_anomaly_shift",
        "dynamic_same_checkpoint_static_only",
    )
    tasks = {}
    passing = 0
    safety = True
    for horizon, _, _ in HORIZONS:
        for threshold in thresholds:
            key = "{:g}".format(threshold)
            reference_results = {}
            task_pass = True
            for reference in references:
                item = comparisons[
                    "dynamic_aligned_minus_" + reference
                ]["all"][horizon][key]
                passed = (
                    item["csi_delta"] is not None
                    and item["csi_delta"] >= minimum_delta
                    and item["average_precision_delta"] is not None
                    and item["average_precision_delta"] > 0
                )
                reference_results[reference] = passed
                task_pass = task_pass and passed
            static_delta = comparisons[
                "dynamic_aligned_minus_static_climatology_trained"
            ]["all"][horizon][key]["csi_delta"]
            if static_delta is not None and static_delta < -minimum_delta:
                safety = False
            task_name = "{}::{}".format(horizon, key)
            tasks[task_name] = {
                "pass": task_pass,
                "references": reference_results,
            }
            passing += int(task_pass)
    return {
        "minimum_csi_delta": minimum_delta,
        "required_passing_tasks": 3,
        "passing_tasks": passing,
        "safety_pass": safety,
        "point_estimate_pass": passing >= 3 and safety,
        "tasks": tasks,
    }


def main():
    args = build_parser().parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",")]
    if len(thresholds) != 2:
        raise ValueError("This locked comparison requires exactly two thresholds.")
    seed_root = Path(args.seed_root)
    cache_path = seed_root / "preconditioning_feature_cache.pt"
    source_summary = seed_root / "preconditioning_probe_summary.json"
    if not cache_path.exists():
        raise FileNotFoundError("Missing feature cache: {}".format(cache_path))
    if not source_summary.exists():
        raise FileNotFoundError("Missing source summary: {}".format(source_summary))
    output = (
        Path(args.output_dir)
        if args.output_dir
        else seed_root / "fair_dynamic_residual_control"
    )
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    caches = torch.load(cache_path, map_location="cpu")
    fit_indices, calibration_indices = split_fit_calibration(
        caches["train"]["cases"], args.calibration_day_fraction
    )
    climatology, feature_diagnostics = attach_features(
        caches, fit_indices, args.pwv_intensity_scale
    )
    probe_kwargs = {
        "radar_channels": caches["train"]["radar"].shape[1],
        "pwv_channels": caches["train"]["pwv_dynamic_residual"].shape[1],
        "hidden_channels": args.hidden_channels,
        "lead_count": caches["train"]["target"].shape[1],
        "threshold_count": len(thresholds),
    }
    probes = {}
    for name in ("static", "dynamic"):
        seed_everything(args.seed)
        probes[name] = ConditionalEventProbe(**probe_kwargs)
    parameter_counts = {
        name: sum(parameter.numel() for parameter in probe.parameters())
        for name, probe in probes.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise AssertionError("Static and dynamic probe parameter counts differ.")
    probes["static"] = train_probe(
        probes["static"],
        caches["train"],
        fit_indices,
        True,
        args,
        "static",
        feature_key="pwv_static_fair",
    )
    probes["dynamic"] = train_probe(
        probes["dynamic"],
        caches["train"],
        fit_indices,
        True,
        args,
        "dynamic",
        feature_key="pwv_dynamic_residual",
    )
    cutoffs = {
        "static": calibrate(
            probes["static"],
            caches["train"],
            calibration_indices,
            True,
            "pwv_static_fair",
            thresholds,
            args,
        ),
        "dynamic": calibrate(
            probes["dynamic"],
            caches["train"],
            calibration_indices,
            True,
            "pwv_dynamic_residual",
            thresholds,
            args,
        ),
    }
    specifications = {
        "static_climatology_trained": (
            "static",
            "pwv_static_fair",
            "real",
        ),
        "dynamic_aligned": (
            "dynamic",
            "pwv_dynamic_residual",
            "real",
        ),
        "dynamic_cross_event": (
            "dynamic",
            "pwv_dynamic_residual",
            "cross_event",
        ),
        "dynamic_spatial_anomaly_shift": (
            "dynamic",
            "pwv_dynamic_spatial_shift",
            "real",
        ),
        "dynamic_tendency_reversed": (
            "dynamic",
            "pwv_dynamic_tendency_reversed",
            "real",
        ),
        "dynamic_same_checkpoint_static_only": (
            "dynamic",
            "pwv_static_fair",
            "real",
        ),
    }
    variants = {}
    for variant, (probe_name, feature_key, control) in specifications.items():
        variants[variant] = evaluate(
            probes[probe_name],
            caches["val"],
            True,
            feature_key,
            control,
            cutoffs[probe_name],
            thresholds,
            args,
        )
    references = tuple(
        name for name in variants if name != "dynamic_aligned"
    )
    comparisons = {
        "dynamic_aligned_minus_" + reference: compare(
            "dynamic_aligned",
            reference,
            variants,
            thresholds,
            args,
        )
        for reference in references
    }
    gate = promotion_gate(
        comparisons, thresholds, args.minimum_csi_delta
    )
    for variant in variants.values():
        variant.pop("eventwise")
    prior = json.loads(source_summary.read_text(encoding="utf-8"))
    summary = sanitize_json_numbers(
        {
            "protocol": "pwv_fair_dynamic_residual_control",
            "source_cache": str(cache_path),
            "source_protocol": prior.get("protocol"),
            "train_samples": len(caches["train"]["radar"]),
            "validation_samples": len(caches["val"]["radar"]),
            "fit_samples": len(fit_indices),
            "calibration_samples": len(calibration_indices),
            "probe_parameters": parameter_counts,
            "feature_contract": [
                "shared static climatology",
                "latest event scalar",
                "mean event scalar",
                "event scalar tendency",
                "latest event spatial anomaly",
                "event spatial-anomaly tendency",
            ],
            "climatology": {
                "source": "fit days only",
                "mean_mm": float(climatology.mean()),
                "spatial_std_mm": float(climatology.std()),
            },
            "feature_diagnostics": feature_diagnostics,
            "calibration_thresholds": cutoffs,
            "variants": variants,
            "paired_comparisons": comparisons,
            "gate": gate,
            "interpretation": (
                "Both trained probes receive the identical static climatology "
                "channel. Only the dynamic probe receives event-specific PWV."
            ),
        }
    )
    summary_path = output / "fair_dynamic_residual_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    safe_torch_save(
        probes["static"].state_dict(), output / "static_probe.ckpt"
    )
    safe_torch_save(
        probes["dynamic"].state_dict(), output / "dynamic_probe.ckpt"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
