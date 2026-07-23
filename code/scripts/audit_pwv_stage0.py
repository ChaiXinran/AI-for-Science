"""Stage-0 audit for the 0--2 h radar/PWV protocol.

The audit is deliberately model-free.  It verifies split isolation, strict
radar/PWV pairing, cadence, heavy-rain support by lead-time bin, and the amount
of independent event support available for later confidence intervals.  It
also writes a train-only spatial PWV climatology used by the signed calibrator.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_horizon_bins(value, frame_minutes):
    bins = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        start_hour, end_hour = [float(part) for part in item.split("-", 1)]
        start = int(round(start_hour * 60.0 / frame_minutes))
        end = int(round(end_hour * 60.0 / frame_minutes))
        if start < 0 or end <= start:
            raise ValueError("Invalid horizon bin: {}".format(item))
        bins.append((item, start, end))
    return bins


def timestamp(path):
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return None


def read_field(path, intensity_scale, invert=True):
    if cv2 is not None:
        pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise ValueError("Could not read {}".format(path))
        pixels = pixels.astype(np.float32)
    else:
        pixels = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if invert:
        pixels = 255.0 - pixels
    return pixels / 255.0 * float(intensity_scale)


def contiguous_window_starts(files, total_length, frame_minutes):
    stamps = [timestamp(path) for path in files]
    starts = []
    cadence_seconds = int(round(frame_minutes * 60.0))
    for start in range(max(0, len(files) - total_length + 1)):
        window = stamps[start : start + total_length]
        if any(item is None for item in window):
            continue
        deltas = [
            int(round((right - left).total_seconds()))
            for left, right in zip(window, window[1:])
        ]
        if all(delta == cadence_seconds for delta in deltas):
            starts.append(start)
    return starts


def safe_corr(left, right):
    if len(left) < 2 or len(right) < 2:
        return None
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if float(left.std()) <= 0.0 or float(right.std()) <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def strict_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_train_climatology(rain_root, pwv_root, manifest, pwv_scale, frame_stride):
    total = None
    total_sq = None
    count = 0
    shape = None
    for relative_dir in manifest["splits"]["train"]:
        rain_paths = sorted((rain_root / relative_dir).glob("*.png"))
        for rain_path in rain_paths[:: max(1, int(frame_stride))]:
            pwv_path = pwv_root / rain_path.relative_to(rain_root)
            if not pwv_path.exists():
                continue
            field = read_field(pwv_path, pwv_scale, invert=True).astype(np.float64)
            if shape is None:
                shape = field.shape
                total = np.zeros(shape, dtype=np.float64)
                total_sq = np.zeros(shape, dtype=np.float64)
            if field.shape != shape:
                raise ValueError(
                    "PWV shape mismatch: {} has {}, expected {}".format(
                        pwv_path, field.shape, shape
                    )
                )
            total += field
            total_sq += field * field
            count += 1
    if count == 0:
        raise ValueError("No paired train PWV frames were available for climatology.")
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count


def split_boundary_report(manifest):
    report = {"overlap": {}, "boundaries": []}
    split_sets = {name: set(items) for name, items in manifest["splits"].items()}
    names = list(split_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = sorted(split_sets[left_name] & split_sets[right_name])
            report["overlap"]["{}_{}".format(left_name, right_name)] = overlap
    for left_name, right_name in (("train", "val"), ("val", "test")):
        left = manifest["splits"].get(left_name, [])
        right = manifest["splits"].get(right_name, [])
        if not left or not right:
            continue
        left_token = "".join(ch for ch in Path(left[-1]).name if ch.isdigit())
        right_token = "".join(ch for ch in Path(right[0]).name if ch.isdigit())
        row = {"left_split": left_name, "right_split": right_name}
        try:
            left_date = datetime.strptime(left_token, "%Y%m%d")
            right_date = datetime.strptime(right_token, "%Y%m%d")
            row.update(
                {
                    "left_last": left[-1],
                    "right_first": right[0],
                    "gap_days": (right_date - left_date).days,
                    "manual_storm_review_required": (right_date - left_date).days <= 1,
                }
            )
        except ValueError:
            row["date_parse_error"] = True
        report["boundaries"].append(row)
    report["has_directory_overlap"] = any(report["overlap"].values())
    return report


def audit_split(
    split,
    relative_dirs,
    rain_root,
    pwv_root,
    climatology,
    thresholds,
    horizons,
    input_length,
    total_length,
    frame_minutes,
    rain_scale,
    pwv_scale,
    transition_samples_per_event,
    feature_windows_per_event,
):
    threshold_rows = {
        "{:g}".format(threshold): {
            label: {
                "positive_windows": 0,
                "positive_target_pixels_across_windows": 0,
                "positive_events": 0,
            }
            for label, _, _ in horizons
        }
        for threshold in thresholds
    }
    missing_pairs = []
    cadence_counts = Counter()
    exact_repeat_pairs = 0
    pwv_transition_pairs = 0
    pwv_abs_change_sum = 0.0
    total_windows = 0
    rejected_noncontiguous = 0
    event_rows = []
    sample_level = []

    for relative_dir in relative_dirs:
        files = sorted((rain_root / relative_dir).glob("*.png"))
        paired = [pwv_root / path.relative_to(rain_root) for path in files]
        for path in paired:
            if not path.exists():
                missing_pairs.append(path.relative_to(pwv_root).as_posix())
        available = [(rain, pwv) for rain, pwv in zip(files, paired) if pwv.exists()]
        if not available:
            event_rows.append(
                {"event": relative_dir, "frames": len(files), "paired_frames": 0, "windows": 0}
            )
            continue
        rain_files = [item[0] for item in available]
        pwv_files = [item[1] for item in available]
        rain = np.stack([read_field(path, rain_scale, invert=True) for path in rain_files])
        first_pwv = read_field(pwv_files[0], pwv_scale, invert=True)
        if rain.shape[1:] != climatology.shape or first_pwv.shape != climatology.shape:
            raise ValueError(
                "Field/climatology shape mismatch in {}: rain={}, pwv={}, climatology={}".format(
                    relative_dir, rain.shape[1:], first_pwv.shape, climatology.shape
                )
            )

        stamps = [timestamp(path) for path in rain_files]
        for left, right in zip(stamps, stamps[1:]):
            if left is not None and right is not None:
                cadence_counts[int(round((right - left).total_seconds()))] += 1
        if len(pwv_files) > 1:
            pair_count = min(len(pwv_files) - 1, max(1, int(transition_samples_per_event)))
            pair_indices = np.unique(
                np.rint(np.linspace(0, len(pwv_files) - 2, num=pair_count)).astype(int)
            )
            for pair_index in pair_indices:
                left = read_field(pwv_files[pair_index], pwv_scale, invert=True)
                right = read_field(pwv_files[pair_index + 1], pwv_scale, invert=True)
                mean_change = float(np.abs(right - left).mean())
                exact_repeat_pairs += int(mean_change == 0.0)
                pwv_transition_pairs += 1
                pwv_abs_change_sum += mean_change

        starts = contiguous_window_starts(rain_files, total_length, frame_minutes)
        possible = max(0, len(rain_files) - total_length + 1)
        rejected_noncontiguous += possible - len(starts)
        total_windows += len(starts)
        event_row = {
            "event": relative_dir,
            "frames": len(files),
            "paired_frames": len(rain_files),
            "windows": len(starts),
            "max_rain_mm_h": float(rain.max()),
            "mean_pwv_mm_sampled": float(first_pwv.mean()),
        }
        event_positive = {
            "{:g}".format(threshold): {label: False for label, _, _ in horizons}
            for threshold in thresholds
        }
        feature_start_set = set()
        if starts:
            feature_count = min(len(starts), max(1, int(feature_windows_per_event)))
            feature_start_set = set(
                starts[index]
                for index in np.unique(
                    np.rint(np.linspace(0, len(starts) - 1, num=feature_count)).astype(int)
                )
            )
        for start in starts:
            collect_features = start in feature_start_set
            if collect_features:
                observed_pwv = np.stack(
                    [
                        read_field(path, pwv_scale, invert=True)
                        for path in pwv_files[start : start + input_length]
                    ]
                )
            else:
                observed_pwv = None
            if collect_features:
                level = float(observed_pwv.mean())
                slope = float((observed_pwv[-1] - observed_pwv[0]).mean())
                anomaly = float(np.abs(observed_pwv.mean(axis=0) - climatology).mean())
                sample_row = {"level": level, "slope": slope, "anomaly_energy": anomaly}
            for threshold in thresholds:
                threshold_key = "{:g}".format(threshold)
                for label, horizon_start, horizon_end in horizons:
                    first = start + input_length + horizon_start
                    last = min(start + input_length + horizon_end, start + total_length)
                    if first >= last:
                        continue
                    target = rain[first:last]
                    positive_pixels = int((target >= threshold).sum())
                    row = threshold_rows[threshold_key][label]
                    row["positive_target_pixels_across_windows"] += positive_pixels
                    if positive_pixels:
                        row["positive_windows"] += 1
                        event_positive[threshold_key][label] = True
                    if collect_features:
                        sample_row[
                            "future_pixels_ge_{}_{}".format(threshold_key, label)
                        ] = positive_pixels
            if collect_features:
                sample_level.append(sample_row)
        for threshold_key, horizon_map in event_positive.items():
            for label, is_positive in horizon_map.items():
                threshold_rows[threshold_key][label]["positive_events"] += int(is_positive)
                event_row["event_positive_ge_{}_{}".format(threshold_key, label)] = bool(is_positive)
        event_rows.append(event_row)

    for threshold_key, horizon_map in threshold_rows.items():
        for label, row in horizon_map.items():
            row["total_windows"] = total_windows
            row["positive_window_rate"] = (
                row["positive_windows"] / total_windows if total_windows else None
            )
            row["total_events"] = len(relative_dirs)
            row["positive_event_rate"] = (
                row["positive_events"] / len(relative_dirs) if relative_dirs else None
            )

    correlations = {}
    for threshold in thresholds:
        threshold_key = "{:g}".format(threshold)
        correlations[threshold_key] = {}
        for label, _, _ in horizons:
            target_key = "future_pixels_ge_{}_{}".format(threshold_key, label)
            target = [row.get(target_key, 0) for row in sample_level]
            correlations[threshold_key][label] = {
                feature: safe_corr([row[feature] for row in sample_level], target)
                for feature in ("level", "slope", "anomaly_energy")
            }

    return {
        "split": split,
        "events": len(relative_dirs),
        "frames": sum(row["frames"] for row in event_rows),
        "paired_frames": sum(row["paired_frames"] for row in event_rows),
        "missing_pair_count": len(missing_pairs),
        "missing_pairs_first_20": missing_pairs[:20],
        "contiguous_windows": total_windows,
        "rejected_noncontiguous_windows": rejected_noncontiguous,
        "radar_cadence_seconds": dict(sorted(cadence_counts.items())),
        "pwv_transition_pairs": pwv_transition_pairs,
        "pwv_transition_sampling": "deterministic_uniform_per_event",
        "pwv_exact_repeat_pairs": exact_repeat_pairs,
        "pwv_exact_repeat_rate": (
            exact_repeat_pairs / pwv_transition_pairs if pwv_transition_pairs else None
        ),
        "pwv_mean_abs_change_mm_per_frame": (
            pwv_abs_change_sum / pwv_transition_pairs if pwv_transition_pairs else None
        ),
        "threshold_support": threshold_rows,
        "window_feature_target_correlations": correlations,
        "event_rows": event_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pwv_root", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=29)
    parser.add_argument("--frame_minutes", type=float, default=6.0)
    parser.add_argument("--thresholds", default="10,20,30")
    parser.add_argument("--horizon_bins", default="0-1,1-2,0-2")
    parser.add_argument("--rain_intensity_scale", type=float, default=35.0)
    parser.add_argument("--pwv_intensity_scale", type=float, default=80.0)
    parser.add_argument("--climatology_frame_stride", type=int, default=6)
    parser.add_argument("--transition_samples_per_event", type=int, default=12)
    parser.add_argument("--feature_windows_per_event", type=int, default=6)
    args = parser.parse_args()

    rain_root = Path(args.data_root).resolve()
    pwv_root = Path(args.pwv_root).resolve()
    manifest_path = Path(args.split_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    thresholds = parse_float_list(args.thresholds)
    horizons = parse_horizon_bins(args.horizon_bins, args.frame_minutes)
    if horizons and max(end for _, _, end in horizons) > args.total_length - args.input_length:
        raise ValueError("A horizon extends beyond the forecast length.")

    climatology, climatology_std, climatology_count = build_train_climatology(
        rain_root,
        pwv_root,
        manifest,
        args.pwv_intensity_scale,
        args.climatology_frame_stride,
    )
    np.savez_compressed(
        output_dir / "pwv_train_climatology.npz",
        mean=climatology,
        std=climatology_std,
        count=np.asarray(climatology_count, dtype=np.int64),
        split_sha256=np.asarray(manifest.get("split_sha256", "")),
        pwv_intensity_scale=np.asarray(args.pwv_intensity_scale, dtype=np.float32),
    )

    report = {
        "protocol": "pwv_signed_calibrator_stage0",
        "data_root": str(rain_root),
        "pwv_root": str(pwv_root),
        "split_manifest": str(manifest_path),
        "split_sha256": manifest.get("split_sha256"),
        "sequence": {
            "input_length": args.input_length,
            "total_length": args.total_length,
            "forecast_length": args.total_length - args.input_length,
            "frame_minutes": args.frame_minutes,
        },
        "thresholds_mm_h": thresholds,
        "horizon_bins": [label for label, _, _ in horizons],
        "split_isolation": split_boundary_report(manifest),
        "climatology": {
            "source_split": "train only",
            "paired_frames": climatology_count,
            "deterministic_frame_stride": args.climatology_frame_stride,
            "shape": list(climatology.shape),
            "mean_mm": float(climatology.mean()),
            "spatial_std_of_mean_mm": float(climatology.std()),
            "mean_temporal_std_mm": float(climatology_std.mean()),
            "artifact": "pwv_train_climatology.npz",
        },
        "station_geometry": {
            "available": False,
            "reason": "Only gridded PWV PNGs are present; station locations/coverage masks are not in the dataset.",
            "required_for_full_protocol": True,
        },
        "splits": {},
    }
    for split, relative_dirs in manifest["splits"].items():
        report["splits"][split] = audit_split(
            split,
            relative_dirs,
            rain_root,
            pwv_root,
            climatology,
            thresholds,
            horizons,
            args.input_length,
            args.total_length,
            args.frame_minutes,
            args.rain_intensity_scale,
            args.pwv_intensity_scale,
            args.transition_samples_per_event,
            args.feature_windows_per_event,
        )

    fingerprint_payload = json.dumps(
        {
            "split_sha256": report["split_sha256"],
            "thresholds": thresholds,
            "horizons": report["horizon_bins"],
            "sequence": report["sequence"],
            "climatology_frame_stride": args.climatology_frame_stride,
            "transition_samples_per_event": args.transition_samples_per_event,
            "feature_windows_per_event": args.feature_windows_per_event,
        },
        sort_keys=True,
    ).encode("utf-8")
    report["audit_sha256"] = hashlib.sha256(fingerprint_payload).hexdigest()
    report = strict_json(report)
    (output_dir / "stage0_support_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "audit_sha256": report["audit_sha256"],
        "split_sha256": report["split_sha256"],
        "climatology": report["climatology"],
        "station_geometry": report["station_geometry"],
        "split_isolation": report["split_isolation"],
        "support": {
            split: row["threshold_support"] for split, row in report["splits"].items()
        },
        "pwv_cadence": {
            split: {
                "exact_repeat_rate": row["pwv_exact_repeat_rate"],
                "mean_abs_change_mm_per_frame": row["pwv_mean_abs_change_mm_per_frame"],
            }
            for split, row in report["splits"].items()
        },
    }
    (output_dir / "stage0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
