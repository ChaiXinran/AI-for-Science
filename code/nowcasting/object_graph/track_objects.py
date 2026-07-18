import math

import numpy as np

from nowcasting.object_graph.feature_extractor import dilate_mask


def mask_iou(mask_a, mask_b):
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(mask_a.sum()) + int(mask_b.sum()) - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def candidate_score(source, target, max_distance_pixels=12.0, sigma_distance=6.0, sigma_intensity=8.0, dilation_radius=2):
    dx = float(target["centroid_x"]) - float(source["centroid_x"])
    dy = float(target["centroid_y"]) - float(source["centroid_y"])
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > max_distance_pixels:
        return None

    iou = mask_iou(dilate_mask(source["mask"], dilation_radius), target["mask"])
    distance_score = math.exp(-(distance * distance) / (2.0 * sigma_distance * sigma_distance))
    area_score = math.exp(-abs(math.log(max(target["area_pixels"], 1) / max(source["area_pixels"], 1))))
    intensity_score = math.exp(
        -abs(float(target["mean_intensity"]) - float(source["mean_intensity"])) / max(float(sigma_intensity), 1e-6)
    )
    score = 0.45 * iou + 0.35 * distance_score + 0.10 * area_score + 0.10 * intensity_score
    return {
        "score": float(score),
        "iou": float(iou),
        "distance": float(distance),
        "area_score": float(area_score),
        "intensity_score": float(intensity_score),
    }


def build_frame_edges(
    source_objects,
    target_objects,
    max_distance_pixels=12.0,
    match_score_threshold=0.25,
    secondary_score_threshold=0.18,
    sigma_distance=6.0,
    sigma_intensity=8.0,
    dilation_radius=2,
):
    """Build one-step object links between adjacent frames."""
    candidates = []
    for source_index, source in enumerate(source_objects):
        for target_index, target in enumerate(target_objects):
            score = candidate_score(
                source,
                target,
                max_distance_pixels=max_distance_pixels,
                sigma_distance=sigma_distance,
                sigma_intensity=sigma_intensity,
                dilation_radius=dilation_radius,
            )
            if score is None or score["score"] < secondary_score_threshold:
                continue
            candidates.append((source_index, target_index, score))

    primary = []
    used_sources = set()
    used_targets = set()
    for source_index, target_index, score in sorted(candidates, key=lambda item: item[2]["score"], reverse=True):
        if score["score"] < match_score_threshold:
            continue
        if source_index in used_sources or target_index in used_targets:
            continue
        primary.append((source_index, target_index, score))
        used_sources.add(source_index)
        used_targets.add(target_index)

    primary_pairs = {(source_index, target_index) for source_index, target_index, _ in primary}
    selected = list(primary)
    for source_index, target_index, score in candidates:
        if (source_index, target_index) in primary_pairs:
            continue
        selected.append((source_index, target_index, score))

    out_degree = {}
    in_degree = {}
    for source_index, target_index, _ in selected:
        out_degree[source_index] = out_degree.get(source_index, 0) + 1
        in_degree[target_index] = in_degree.get(target_index, 0) + 1

    edges = []
    for source_index, target_index, score in selected:
        relation = "continue"
        if out_degree.get(source_index, 0) >= 2 and in_degree.get(target_index, 0) >= 2:
            relation = "reorganize"
        elif out_degree.get(source_index, 0) >= 2:
            relation = "split"
        elif in_degree.get(target_index, 0) >= 2:
            relation = "merge"
        edges.append(
            {
                "source_index": source_index,
                "target_index": target_index,
                "source_object_id": source_objects[source_index]["object_id"],
                "target_object_id": target_objects[target_index]["object_id"],
                "source_time": source_objects[source_index]["frame_time"],
                "target_time": target_objects[target_index]["frame_time"],
                "source_frame_index": source_objects[source_index]["frame_index"],
                "target_frame_index": target_objects[target_index]["frame_index"],
                "match_score": score["score"],
                "match_iou": score["iou"],
                "centroid_distance": score["distance"],
                "relation": relation,
                "is_primary": int((source_index, target_index) in primary_pairs),
            }
        )
    return edges, primary_pairs


def assign_tracks(objects_by_frame, edges_by_step):
    """Assign simple track ids from primary one-to-one links and enrich motion features."""
    next_track_id = 0
    by_id = {}
    for objects in objects_by_frame:
        for obj in objects:
            by_id[obj["object_id"]] = obj

    for frame_index, objects in enumerate(objects_by_frame):
        if frame_index == 0:
            for obj in objects:
                obj["track_id"] = next_track_id
                obj["object_age"] = 1
                next_track_id += 1
            continue

        inherited = set()
        for edge in edges_by_step.get(frame_index - 1, []):
            if not edge["is_primary"]:
                continue
            source = by_id[edge["source_object_id"]]
            target = by_id[edge["target_object_id"]]
            target["track_id"] = source["track_id"]
            target["object_age"] = int(source.get("object_age", 1)) + 1
            inherited.add(target["object_id"])

        for obj in objects:
            if obj["object_id"] in inherited:
                continue
            obj["track_id"] = next_track_id
            obj["object_age"] = 1
            next_track_id += 1

    previous_by_track = {}
    for objects in objects_by_frame:
        for obj in objects:
            prev = previous_by_track.get(obj["track_id"])
            if prev is None:
                obj["velocity_x"] = 0.0
                obj["velocity_y"] = 0.0
                obj["area_change"] = 0.0
                obj["intensity_change"] = 0.0
            else:
                obj["velocity_x"] = float(obj["centroid_x"]) - float(prev["centroid_x"])
                obj["velocity_y"] = float(obj["centroid_y"]) - float(prev["centroid_y"])
                obj["area_change"] = math.log(max(obj["area_pixels"], 1) / max(prev["area_pixels"], 1))
                obj["intensity_change"] = float(obj["mean_intensity"]) - float(prev["mean_intensity"])
            previous_by_track[obj["track_id"]] = obj


def add_one_step_labels(objects_by_frame, edges_by_step):
    by_id = {}
    for objects in objects_by_frame:
        for obj in objects:
            by_id[obj["object_id"]] = obj
            obj["label_valid_next"] = 0
            obj["label_survive"] = 0
            obj["label_decay"] = 0
            obj["label_split"] = 0
            obj["label_merge"] = 0
            obj["label_initiation"] = 0

    for frame_index, objects in enumerate(objects_by_frame):
        if frame_index > 0:
            incoming = [edge for edge in edges_by_step.get(frame_index - 1, []) if edge["target_object_id"] in by_id]
            target_ids = {edge["target_object_id"] for edge in incoming}
            for obj in objects:
                obj["label_initiation"] = int(obj["object_id"] not in target_ids)

        outgoing = [edge for edge in edges_by_step.get(frame_index, []) if edge["source_object_id"] in by_id]
        by_source = {}
        for edge in outgoing:
            by_source.setdefault(edge["source_object_id"], []).append(edge)
        for obj in objects:
            obj["label_valid_next"] = int(frame_index < len(objects_by_frame) - 1)
            if not obj["label_valid_next"]:
                continue
            edges = by_source.get(obj["object_id"], [])
            obj["label_survive"] = int(len(edges) > 0)
            obj["label_decay"] = int(len(edges) == 0)
            obj["label_split"] = int(any(edge["relation"] in ("split", "reorganize") for edge in edges))
            obj["label_merge"] = int(any(edge["relation"] in ("merge", "reorganize") for edge in edges))
