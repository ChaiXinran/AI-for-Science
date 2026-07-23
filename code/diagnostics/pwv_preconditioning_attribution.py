"""Attribute causal-PWV probe skill without retraining any model.

The observed causal history is decomposed into:

* a fit-day static spatial climatology;
* an event-level spatially constant moisture offset;
* a residual event-specific spatial anomaly.

All interventions are evaluated through the same trained long-PWV checkpoint
and the same frozen calibration thresholds.
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
)
from diagnostics.pwv_preconditioning_probe import (  # noqa: E402
    PRIMARY_STRATUM,
    attach_long_features,
    causal_pwv_features,
    evaluate,
)
from nowcasting.experiments.common import (  # noqa: E402
    load_model_state,
    sanitize_json_numbers,
    seed_everything,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Same-checkpoint attribution for causal PWV preconditioning"
    )
    parser.add_argument("--seed_root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--thresholds", default="10,20")
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--histogram_bins", type=int, default=1000)
    parser.add_argument("--pwv_intensity_scale", type=float, default=80.0)
    parser.add_argument("--calibration_day_fraction", type=float, default=0.2)
    parser.add_argument("--minimum_csi_delta", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def decompose_pwv_history(history, climatology, scale):
    """Return physically interpretable PWV history interventions.

    Spatial statistics exclude the zero-padded area surrounding the original
    70x66 domain on the 96x96 model canvas.
    """
    history = history.float()
    climatology = climatology.float()
    valid = (climatology > 1e-6).to(history.dtype)
    valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    global_climatology = (
        (climatology * valid).sum(dim=(-2, -1), keepdim=True)
        / valid_count
    )
    residual = (history - climatology) * valid
    scalar_offset = (
        residual.sum(dim=(-2, -1), keepdim=True) / valid_count
    )
    spatial_anomaly = (residual - scalar_offset * valid) * valid

    static = climatology.expand_as(history)
    static_plus_scalar = climatology + scalar_offset * valid
    scalar_only_reference = global_climatology * valid
    scalar_only = scalar_only_reference + scalar_offset * valid
    anomaly_only = scalar_only_reference + spatial_anomaly
    variants = {
        "pwv_real": history,
        "pwv_static_climatology": static,
        "pwv_static_plus_event_scalar": static_plus_scalar,
        "pwv_event_scalar_only": scalar_only,
        "pwv_event_spatial_anomaly_only": anomaly_only,
    }
    variants = {
        key: value.clamp(0.0, float(scale))
        for key, value in variants.items()
    }
    references = {
        "pwv_real": climatology,
        "pwv_static_climatology": climatology,
        "pwv_static_plus_event_scalar": climatology,
        "pwv_event_scalar_only": scalar_only_reference,
        "pwv_event_spatial_anomaly_only": scalar_only_reference,
    }
    diagnostics = {
        "valid_fraction": float(valid.mean()),
        "global_climatology_mm": float(global_climatology.mean()),
        "mean_abs_event_scalar_mm": float(scalar_offset.abs().mean()),
        "mean_event_spatial_anomaly_std_mm": float(
            spatial_anomaly.std(dim=(-2, -1), unbiased=False).mean()
        ),
    }
    return variants, references, diagnostics


def attach_attribution_features(cache, climatology, scale):
    histories, references, diagnostics = decompose_pwv_history(
        cache["pwv_history"], climatology, scale
    )
    for key, history in histories.items():
        cache[key] = causal_pwv_features(
            history,
            references[key],
            scale,
        ).half()
    return diagnostics


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


def evidence_gate(comparisons, thresholds, minimum_delta):
    tasks = {}
    passing = 0
    comparison = comparisons[
        "pwv_real_minus_pwv_static_plus_event_scalar"
    ]
    for horizon, _, _ in HORIZONS:
        for threshold in thresholds:
            key = "{:g}".format(threshold)
            item = comparison["all"][horizon][key]
            passed = (
                item["csi_delta"] is not None
                and item["csi_delta"] >= minimum_delta
                and item["average_precision_delta"] is not None
                and item["average_precision_delta"] > 0
            )
            tasks["{}::{}".format(horizon, key)] = passed
            passing += int(passed)
    return {
        "question": (
            "Do event-specific spatial PWV anomalies add skill beyond static "
            "geography plus the event-level scalar moisture regime?"
        ),
        "minimum_csi_delta": minimum_delta,
        "required_passing_tasks": 3,
        "passing_tasks": passing,
        "point_estimate_pass": passing >= 3,
        "tasks": tasks,
    }


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    seed_root = Path(args.seed_root)
    summary_path = seed_root / "preconditioning_probe_summary.json"
    cache_path = seed_root / "preconditioning_feature_cache.pt"
    long_checkpoint = seed_root / "long_probe.ckpt"
    for path in (summary_path, cache_path, long_checkpoint):
        if not path.exists():
            raise FileNotFoundError("Required attribution input missing: {}".format(path))

    prior = json.loads(summary_path.read_text(encoding="utf-8"))
    thresholds = [float(item) for item in args.thresholds.split(",")]
    caches = torch.load(cache_path, map_location="cpu")
    fit_indices, _ = split_fit_calibration(
        caches["train"]["cases"], args.calibration_day_fraction
    )
    climatology = caches["train"]["pwv_history"][fit_indices].float().mean(
        dim=(0, 1), keepdim=True
    )
    # Preserve the exact real feature construction used by the trained probe.
    attach_long_features(caches, fit_indices, args)
    decomposition = attach_attribution_features(
        caches["val"], climatology, args.pwv_intensity_scale
    )
    real_feature_error = float(
        (
            caches["val"]["pwv_real"].float()
            - caches["val"]["pwv_long"].float()
        )
        .abs()
        .max()
    )
    if real_feature_error > 1e-3:
        raise AssertionError(
            "Reconstructed real-PWV features disagree with the trained representation."
        )

    probe_kwargs = {
        "radar_channels": caches["val"]["radar"].shape[1],
        "pwv_channels": caches["val"]["pwv_real"].shape[1],
        "hidden_channels": args.hidden_channels,
        "lead_count": caches["val"]["target"].shape[1],
        "threshold_count": len(thresholds),
    }
    probe = ConditionalEventProbe(**probe_kwargs)
    probe.load_state_dict(
        load_model_state(long_checkpoint, "cpu"),
        strict=True,
    )
    cutoffs = prior["calibration_thresholds"]["long"]
    controls = (
        "pwv_real",
        "pwv_static_climatology",
        "pwv_static_plus_event_scalar",
        "pwv_event_scalar_only",
        "pwv_event_spatial_anomaly_only",
    )
    variants = {
        name: evaluate(
            probe,
            caches["val"],
            True,
            name,
            name.removeprefix("pwv_"),
            cutoffs,
            thresholds,
            args,
        )
        for name in controls
    }
    pairs = (
        ("pwv_real", "pwv_static_climatology"),
        ("pwv_static_plus_event_scalar", "pwv_static_climatology"),
        ("pwv_real", "pwv_static_plus_event_scalar"),
        ("pwv_real", "pwv_event_scalar_only"),
        ("pwv_real", "pwv_event_spatial_anomaly_only"),
    )
    comparisons = {
        "{}_minus_{}".format(left, right): compare(
            left, right, variants, thresholds, args
        )
        for left, right in pairs
    }
    gate = evidence_gate(
        comparisons, thresholds, args.minimum_csi_delta
    )
    for variant in variants.values():
        variant.pop("eventwise")
    output = (
        Path(args.output)
        if args.output
        else seed_root / "preconditioning_attribution_summary.json"
    )
    payload = sanitize_json_numbers(
        {
            "protocol": "pwv_preconditioning_same_checkpoint_attribution",
            "source_summary": str(summary_path),
            "source_checkpoint": str(long_checkpoint),
            "source_cache": str(cache_path),
            "samples": len(caches["val"]["radar"]),
            "probe_parameters": sum(
                parameter.numel() for parameter in probe.parameters()
            ),
            "decomposition": {
                "static_climatology": "fit-day mean PWV field",
                "event_scalar": (
                    "per-anchor valid-domain mean departure from climatology"
                ),
                "event_spatial_anomaly": (
                    "per-anchor residual after removing climatology and the "
                    "event scalar; zero mean over the valid domain"
                ),
                **decomposition,
                "real_feature_max_abs_error": real_feature_error,
            },
            "calibration_thresholds": cutoffs,
            "variants": variants,
            "paired_comparisons": comparisons,
            "spatial_anomaly_gate": gate,
            "interpretation": (
                "Every PWV variant uses the same long-PWV checkpoint and "
                "frozen calibration thresholds. No model is retrained."
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
