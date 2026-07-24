"""Object-level failure attribution for radar-only precipitation nowcasts.

The diagnostic is deliberately model-agnostic.  It decomposes observed and
forecast precipitation objects into translation, growth, decay, birth, and
split/merge regimes, then estimates upper-bound CSI gains from displacement,
intensity, and existence oracles.  Only the final observed input frame and
forecast-period fields are used; no future information is exposed to a model.
"""

import math

import numpy as np

from nowcasting.object_graph.extract_objects import extract_frame_objects
from nowcasting.object_graph.track_objects import (
    assign_tracks,
    build_frame_edges,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    linear_sum_assignment = None


REGIMES = ("translation", "rapid_growth", "rapid_decay", "birth", "split_merge")
ORACLES = ("original", "displacement", "intensity", "birth_existence", "existence")


def _threshold_key(value):
    return "{:g}".format(float(value))


def _horizon_indices(length, frame_minutes, horizon_bins):
    result = {}
    for start_hour, end_hour, label in horizon_bins:
        indices = [
            index
            for index in range(length)
            if float(start_hour) * 60.0 < (index + 1) * frame_minutes <= float(end_hour) * 60.0
        ]
        if indices:
            result[label] = indices
    return result


def _prepare_sequence(fields, threshold, min_area, tracking):
    objects_by_frame = []
    next_id = 0
    for frame_index, field in enumerate(fields):
        objects = extract_frame_objects(field, threshold, min_area, grid_km=1.0)
        for obj in objects:
            obj["object_id"] = next_id
            obj["frame_index"] = frame_index
            obj["frame_time"] = str(frame_index)
            next_id += 1
        objects_by_frame.append(objects)

    edges_by_step = {}
    for frame_index in range(max(len(objects_by_frame) - 1, 0)):
        edges, _ = build_frame_edges(
            objects_by_frame[frame_index],
            objects_by_frame[frame_index + 1],
            **tracking
        )
        edges_by_step[frame_index] = edges
    assign_tracks(objects_by_frame, edges_by_step)
    return objects_by_frame, edges_by_step


def _relative_change(current, previous, key):
    denominator = max(abs(float(previous[key])), 1e-6)
    return (float(current[key]) - float(previous[key])) / denominator


def classify_regimes(objects_by_frame, edges_by_step, change_fraction):
    """Attach a mutually exclusive regime label to every post-issue object."""
    by_id = {
        obj["object_id"]: obj
        for objects in objects_by_frame
        for obj in objects
    }
    incoming = {}
    for edges in edges_by_step.values():
        for edge in edges:
            incoming.setdefault(edge["target_object_id"], []).append(edge)

    for frame_index, objects in enumerate(objects_by_frame):
        for obj in objects:
            if frame_index == 0:
                obj["regime"] = "translation"
                continue
            parents = incoming.get(obj["object_id"], [])
            if not parents:
                obj["regime"] = "birth"
                continue
            if any(edge["relation"] in ("split", "merge", "reorganize") for edge in parents):
                obj["regime"] = "split_merge"
                continue
            primary = [edge for edge in parents if edge["is_primary"]]
            parent_edge = primary[0] if primary else max(parents, key=lambda item: item["match_score"])
            parent = by_id[parent_edge["source_object_id"]]
            changes = (
                _relative_change(obj, parent, "area_pixels"),
                _relative_change(obj, parent, "mean_intensity"),
                _relative_change(obj, parent, "max_intensity"),
            )
            if max(changes) >= change_fraction:
                obj["regime"] = "rapid_growth"
            elif min(changes) <= -change_fraction:
                obj["regime"] = "rapid_decay"
            else:
                obj["regime"] = "translation"


def _pair_score(pred_obj, target_obj, distance_sigma):
    intersection = int(np.logical_and(pred_obj["mask"], target_obj["mask"]).sum())
    union = int(pred_obj["mask"].sum()) + int(target_obj["mask"].sum()) - intersection
    iou = float(intersection) / float(union) if union else 0.0
    dx = float(pred_obj["centroid_x"]) - float(target_obj["centroid_x"])
    dy = float(pred_obj["centroid_y"]) - float(target_obj["centroid_y"])
    distance = math.sqrt(dx * dx + dy * dy)
    distance_score = math.exp(-(distance * distance) / (2.0 * distance_sigma * distance_sigma))
    return 0.65 * iou + 0.35 * distance_score, iou, distance


def pair_objects(
    pred_objects,
    target_objects,
    iou_threshold=0.1,
    max_distance_pixels=12.0,
    distance_sigma=6.0,
):
    """Associate forecast and observed objects using Hungarian assignment."""
    if not pred_objects or not target_objects:
        return [], set(), set()
    scores = np.full((len(pred_objects), len(target_objects)), -1.0e6, dtype="float64")
    details = {}
    for pred_index, pred_obj in enumerate(pred_objects):
        for target_index, target_obj in enumerate(target_objects):
            score, iou, distance = _pair_score(pred_obj, target_obj, distance_sigma)
            if iou < iou_threshold and distance > max_distance_pixels:
                continue
            scores[pred_index, target_index] = score
            details[(pred_index, target_index)] = (iou, distance, score)

    if linear_sum_assignment is not None:
        row_indices, col_indices = linear_sum_assignment(-scores)
        pairs = [
            (int(row), int(col), *details[(int(row), int(col))])
            for row, col in zip(row_indices, col_indices)
            if (int(row), int(col)) in details
        ]
    else:
        candidates = sorted(
            (
                (score, pred_index, target_index)
                for (pred_index, target_index), (_, _, score) in details.items()
            ),
            reverse=True,
        )
        used_pred = set()
        used_target = set()
        pairs = []
        for _, pred_index, target_index in candidates:
            if pred_index in used_pred or target_index in used_target:
                continue
            iou, distance, score = details[(pred_index, target_index)]
            pairs.append((pred_index, target_index, iou, distance, score))
            used_pred.add(pred_index)
            used_target.add(target_index)

    matched_pred = {pair[0] for pair in pairs}
    matched_target = {pair[1] for pair in pairs}
    return pairs, matched_pred, matched_target


def _shift_zero(values, dx, dy):
    shifted = np.zeros_like(values)
    height, width = values.shape
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return shifted


def oracle_fields(pred_field, target_field, pred_objects, target_objects, pairs, matched_pred, matched_target):
    displacement = np.array(pred_field, copy=True)
    intensity = np.array(pred_field, copy=True)
    existence = np.array(pred_field, copy=True)
    birth_existence = np.array(pred_field, copy=True)

    for pred_index, target_index, _, _, _ in pairs:
        pred_obj = pred_objects[pred_index]
        target_obj = target_objects[target_index]
        source_values = np.where(pred_obj["mask"], pred_field, 0.0)
        dx = int(round(float(target_obj["centroid_x"]) - float(pred_obj["centroid_x"])))
        dy = int(round(float(target_obj["centroid_y"]) - float(pred_obj["centroid_y"])))
        displacement[pred_obj["mask"]] = 0.0
        displacement = np.maximum(displacement, _shift_zero(source_values, dx, dy))

        scale = float(target_obj["intensity_sum"]) / max(float(pred_obj["intensity_sum"]), 1e-6)
        intensity[pred_obj["mask"]] = np.maximum(0.0, pred_field[pred_obj["mask"]] * scale)

    for pred_index, pred_obj in enumerate(pred_objects):
        if pred_index not in matched_pred:
            existence[pred_obj["mask"]] = 0.0
    for target_index, target_obj in enumerate(target_objects):
        if target_index in matched_target:
            continue
        existence[target_obj["mask"]] = target_field[target_obj["mask"]]
        if target_obj.get("regime") == "birth":
            birth_existence[target_obj["mask"]] = target_field[target_obj["mask"]]

    return {
        "original": np.asarray(pred_field),
        "displacement": displacement,
        "intensity": intensity,
        "birth_existence": birth_existence,
        "existence": existence,
    }


def event_counts(pred_field, target_field, threshold):
    pred_event = np.asarray(pred_field) >= threshold
    target_event = np.asarray(target_field) >= threshold
    return {
        "hit": int(np.logical_and(pred_event, target_event).sum()),
        "miss": int(np.logical_and(~pred_event, target_event).sum()),
        "false_alarm": int(np.logical_and(pred_event, ~target_event).sum()),
    }


def _empty_regime_counts():
    return {
        regime: {
            "observed": 0,
            "predicted": 0,
            "detected": 0,
            "correct_type": 0,
            "missed": 0,
            "false_alarm": 0,
        }
        for regime in REGIMES
    }


def _empty_oracle_counts():
    return {name: {"hit": 0, "miss": 0, "false_alarm": 0} for name in ORACLES}


def analyze_sample(
    last_input,
    pred,
    target,
    thresholds=(10.0, 20.0, 30.0),
    change_fractions=(0.2, 0.4, 0.6),
    frame_minutes=6.0,
    horizon_bins=((0.0, 1.0, "0-1h"), (1.0, 2.0, "1-2h")),
    min_area=4,
    iou_threshold=0.1,
    max_distance_pixels=12.0,
    tracking_threshold_ratio=0.5,
    tracking=None,
):
    tracking = dict(tracking or {})
    tracking.setdefault("max_distance_pixels", max_distance_pixels)
    tracking.setdefault("match_score_threshold", 0.25)
    tracking.setdefault("secondary_score_threshold", 0.18)
    tracking.setdefault("sigma_distance", 6.0)
    tracking.setdefault("sigma_intensity", 8.0)
    tracking.setdefault("dilation_radius", 2)
    horizons = _horizon_indices(pred.shape[0], frame_minutes, horizon_bins)
    record = {"thresholds": {}}

    for threshold in thresholds:
        threshold_result = {}
        truth_fields = np.concatenate([last_input[None], target], axis=0)
        pred_fields = np.concatenate([last_input[None], pred], axis=0)
        tracking_threshold = max(float(threshold) * float(tracking_threshold_ratio), 1e-6)
        truth_objects, truth_edges = _prepare_sequence(
            truth_fields, tracking_threshold, min_area, tracking
        )
        pred_objects, pred_edges = _prepare_sequence(
            pred_fields, tracking_threshold, min_area, tracking
        )
        for change_fraction in change_fractions:
            classify_regimes(truth_objects, truth_edges, change_fraction)
            classify_regimes(pred_objects, pred_edges, change_fraction)
            by_horizon = {}
            for horizon, indices in horizons.items():
                regime_counts = _empty_regime_counts()
                oracle_counts = _empty_oracle_counts()
                for lead_index in indices:
                    observed = [
                        obj
                        for obj in truth_objects[lead_index + 1]
                        if float(obj["max_intensity"]) >= float(threshold)
                    ]
                    forecast = [
                        obj
                        for obj in pred_objects[lead_index + 1]
                        if float(obj["max_intensity"]) >= tracking_threshold
                    ]
                    active_forecast_indices = {
                        index
                        for index, obj in enumerate(forecast)
                        if float(obj["max_intensity"]) >= float(threshold)
                    }
                    pairs, matched_pred, matched_target = pair_objects(
                        forecast,
                        observed,
                        iou_threshold=iou_threshold,
                        max_distance_pixels=max_distance_pixels,
                    )
                    for obj in observed:
                        regime_counts[obj["regime"]]["observed"] += 1
                    for pred_index in active_forecast_indices:
                        regime_counts[forecast[pred_index]["regime"]]["predicted"] += 1
                    for pred_index, target_index, _, _, _ in pairs:
                        if pred_index not in active_forecast_indices:
                            continue
                        pred_regime = forecast[pred_index]["regime"]
                        target_regime = observed[target_index]["regime"]
                        regime_counts[target_regime]["detected"] += 1
                        if pred_regime == target_regime:
                            regime_counts[target_regime]["correct_type"] += 1
                    active_matched_target = {
                        target_index
                        for pred_index, target_index, _, _, _ in pairs
                        if pred_index in active_forecast_indices
                    }
                    for target_index, obj in enumerate(observed):
                        if target_index not in active_matched_target:
                            regime_counts[obj["regime"]]["missed"] += 1
                    for pred_index, obj in enumerate(forecast):
                        if pred_index in active_forecast_indices and pred_index not in matched_pred:
                            regime_counts[obj["regime"]]["false_alarm"] += 1

                    oracle_map = oracle_fields(
                        pred[lead_index],
                        target[lead_index],
                        forecast,
                        observed,
                        pairs,
                        matched_pred,
                        matched_target,
                    )
                    for name, field in oracle_map.items():
                        counts = event_counts(field, target[lead_index], threshold)
                        for key, value in counts.items():
                            oracle_counts[name][key] += value
                by_horizon[horizon] = {
                    "regimes": regime_counts,
                    "oracles": oracle_counts,
                }
            threshold_result[_threshold_key(change_fraction)] = by_horizon
        record["thresholds"][_threshold_key(threshold)] = threshold_result
    return record


def _sum_record_tree(destination, source):
    for key, value in source.items():
        if isinstance(value, dict):
            _sum_record_tree(destination.setdefault(key, {}), value)
        elif isinstance(value, (int, float)):
            destination[key] = destination.get(key, 0) + value


def _safe_divide(a, b):
    return float(a) / float(b) if b else None


def _finalize_summary(summed):
    result = {"thresholds": {}}
    for threshold, by_change in summed.get("thresholds", {}).items():
        result["thresholds"][threshold] = {}
        for change, by_horizon in by_change.items():
            result["thresholds"][threshold][change] = {}
            for horizon, values in by_horizon.items():
                regimes = {}
                for regime, counts in values["regimes"].items():
                    observed = counts.get("observed", 0)
                    predicted = counts.get("predicted", 0)
                    correct = counts.get("correct_type", 0)
                    strict_den = observed + predicted - correct
                    regimes[regime] = {
                        **counts,
                        "object_pod": _safe_divide(counts.get("detected", 0), observed),
                        "strict_type_pod": _safe_divide(correct, observed),
                        "object_far": _safe_divide(counts.get("false_alarm", 0), predicted),
                        "strict_type_csi": _safe_divide(correct, strict_den),
                    }
                oracles = {}
                original_csi = None
                for name, counts in values["oracles"].items():
                    denominator = counts["hit"] + counts["miss"] + counts["false_alarm"]
                    csi = _safe_divide(counts["hit"], denominator)
                    if name == "original":
                        original_csi = csi
                    oracles[name] = {**counts, "csi": csi}
                for name, metrics in oracles.items():
                    metrics["csi_delta_vs_original"] = (
                        None
                        if metrics["csi"] is None or original_csi is None
                        else metrics["csi"] - original_csi
                    )
                result["thresholds"][threshold][change][horizon] = {
                    "regimes": regimes,
                    "oracles": oracles,
                }
    return result


def summarize_records(records):
    summed = {}
    for record in records:
        _sum_record_tree(summed, record)
    return _finalize_summary(summed)


def cluster_bootstrap(records, case_names, iterations=500, seed=2026):
    """Bootstrap CSI oracle deltas by case/day clusters."""
    if iterations <= 0 or not records:
        return {}
    groups = {}
    for record, case_name in zip(records, case_names):
        groups.setdefault(str(case_name), []).append(record)
    group_names = sorted(groups)
    if not group_names:
        return {}
    rng = np.random.default_rng(seed)
    samples = {}
    for _ in range(iterations):
        selected = rng.choice(group_names, size=len(group_names), replace=True)
        sampled_records = [
            record
            for group_name in selected
            for record in groups[group_name]
        ]
        summary = summarize_records(sampled_records)
        for threshold, by_change in summary["thresholds"].items():
            for change, by_horizon in by_change.items():
                for horizon, values in by_horizon.items():
                    for oracle, metrics in values["oracles"].items():
                        if oracle == "original" or metrics["csi_delta_vs_original"] is None:
                            continue
                        key = (threshold, change, horizon, oracle)
                        samples.setdefault(key, []).append(metrics["csi_delta_vs_original"])
    result = {}
    for (threshold, change, horizon, oracle), values in samples.items():
        result.setdefault(threshold, {}).setdefault(change, {}).setdefault(horizon, {})[oracle] = {
            "mean": float(np.mean(values)),
            "low_95": float(np.quantile(values, 0.025)),
            "high_95": float(np.quantile(values, 0.975)),
            "iterations": len(values),
            "clusters": len(group_names),
        }
    return result
