import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nowcasting.object_graph.extract_objects import extract_frame_objects
from nowcasting.object_graph.feature_extractor import add_front_pwv_features, add_pwv_features
from nowcasting.object_graph.image_io import find_event_dirs, read_grayscale_frame, relative_event_id
from nowcasting.object_graph.track_objects import add_one_step_labels, assign_tracks, build_frame_edges


OBJECT_FIELDS = [
    "event_id",
    "frame_index",
    "frame_time",
    "object_id",
    "track_id",
    "local_id",
    "centroid_x",
    "centroid_y",
    "area_pixels",
    "area_km2",
    "mean_intensity",
    "max_intensity",
    "intensity_sum",
    "perimeter_pixels",
    "aspect_ratio",
    "eccentricity",
    "velocity_x",
    "velocity_y",
    "area_change",
    "intensity_change",
    "object_age",
    "pwv_inside_mean",
    "pwv_inside_max",
    "pwv_inside_std",
    "pwv_ring_mean",
    "pwv_ring_max",
    "pwv_ring_std",
    "pwv_inner_outer_diff",
    "pwv_front_mean",
    "pwv_delta_2",
    "pwv_delta_5",
    "pwv_delta_10",
    "pwv_gradient_x",
    "pwv_gradient_y",
    "pwv_gradient_parallel",
    "label_valid_next",
    "label_survive",
    "label_decay",
    "label_split",
    "label_merge",
    "label_initiation",
]

EDGE_FIELDS = [
    "event_id",
    "source_object_id",
    "target_object_id",
    "source_frame_index",
    "target_frame_index",
    "source_time",
    "target_time",
    "match_score",
    "match_iou",
    "centroid_distance",
    "relation",
    "is_primary",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Build offline Oracle object-graph tables from calibrated PNG frames.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pwv_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="../outputs/object_graph_pilot")
    parser.add_argument("--img_height", type=int, default=96)
    parser.add_argument("--img_width", type=int, default=96)
    parser.add_argument("--intensity_scale", type=float, default=35.0)
    parser.add_argument("--pixel_min", type=float, default=0.0)
    parser.add_argument("--pixel_max", type=float, default=255.0)
    parser.add_argument("--no_invert", action="store_true")
    parser.add_argument("--pwv_intensity_scale", type=float, default=80.0)
    parser.add_argument("--pwv_pixel_min", type=float, default=0.0)
    parser.add_argument("--pwv_pixel_max", type=float, default=255.0)
    parser.add_argument("--pwv_invert", action="store_true")
    parser.add_argument("--threshold", type=float, default=16.0)
    parser.add_argument("--min_area", type=int, default=4)
    parser.add_argument("--grid_km", type=float, default=1.0)
    parser.add_argument("--ring_radius_pixels", type=int, default=5)
    parser.add_argument("--front_distance_pixels", type=int, default=8)
    parser.add_argument("--front_radius_pixels", type=int, default=4)
    parser.add_argument("--max_distance_pixels", type=float, default=12.0)
    parser.add_argument("--match_score_threshold", type=float, default=0.25)
    parser.add_argument("--secondary_score_threshold", type=float, default=0.18)
    parser.add_argument("--sigma_distance", type=float, default=6.0)
    parser.add_argument("--sigma_intensity", type=float, default=8.0)
    parser.add_argument("--dilation_radius", type=int, default=2)
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--max_frames_per_event", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    return parser


def read_event_frames(event_dir, data_root, args):
    radar_paths = sorted(Path(event_dir).glob("*.png"))
    if args.max_frames_per_event > 0:
        radar_paths = radar_paths[: args.max_frames_per_event]
    radar_frames = [
        read_grayscale_frame(
            path,
            args.img_height,
            args.img_width,
            args.intensity_scale,
            args.pixel_min,
            args.pixel_max,
            not args.no_invert,
        )
        for path in radar_paths
    ]
    pwv_frames = None
    if args.pwv_root:
        pwv_frames = []
        for radar_path in radar_paths:
            rel = radar_path.relative_to(Path(data_root))
            pwv_path = Path(args.pwv_root) / rel
            if pwv_path.exists():
                pwv_frames.append(
                    read_grayscale_frame(
                        pwv_path,
                        args.img_height,
                        args.img_width,
                        args.pwv_intensity_scale,
                        args.pwv_pixel_min,
                        args.pwv_pixel_max,
                        args.pwv_invert,
                    )
                )
            else:
                pwv_frames.append(np.zeros((args.img_height, args.img_width), dtype="float32"))
    return radar_paths, radar_frames, pwv_frames


def process_event(event_dir, data_root, args, object_id_start):
    event_id = relative_event_id(event_dir, data_root)
    radar_paths, radar_frames, pwv_frames = read_event_frames(event_dir, data_root, args)
    objects_by_frame = []
    next_object_id = object_id_start
    for frame_index, field in enumerate(radar_frames):
        objects = extract_frame_objects(field, args.threshold, args.min_area, args.grid_km)
        for obj in objects:
            obj["event_id"] = event_id
            obj["frame_index"] = frame_index
            obj["frame_time"] = radar_paths[frame_index].stem
            obj["object_id"] = next_object_id
            next_object_id += 1
        objects_by_frame.append(objects)

    add_pwv_features(objects_by_frame, pwv_frames, args.ring_radius_pixels, lag_frames=(2, 5, 10))

    edges_by_step = {}
    for frame_index in range(max(len(objects_by_frame) - 1, 0)):
        edges, _ = build_frame_edges(
            objects_by_frame[frame_index],
            objects_by_frame[frame_index + 1],
            max_distance_pixels=args.max_distance_pixels,
            match_score_threshold=args.match_score_threshold,
            secondary_score_threshold=args.secondary_score_threshold,
            sigma_distance=args.sigma_distance,
            sigma_intensity=args.sigma_intensity,
            dilation_radius=args.dilation_radius,
        )
        for edge in edges:
            edge["event_id"] = event_id
        edges_by_step[frame_index] = edges

    assign_tracks(objects_by_frame, edges_by_step)
    add_front_pwv_features(objects_by_frame, pwv_frames, args.front_distance_pixels, args.front_radius_pixels)
    add_one_step_labels(objects_by_frame, edges_by_step)
    objects = [obj for frame_objects in objects_by_frame for obj in frame_objects]
    edges = [edge for step_edges in edges_by_step.values() for edge in step_edges]
    return objects, edges, next_object_id


def split_events(event_ids, train_ratio, val_ratio):
    if len(event_ids) == 1:
        return {event_ids[0]: "train"}
    if len(event_ids) == 2:
        return {event_ids[0]: "train", event_ids[1]: "test"}
    train_end = int(len(event_ids) * train_ratio)
    val_end = train_end + int(len(event_ids) * val_ratio)
    train_end = max(1, train_end)
    val_end = max(train_end, val_end)
    if val_end >= len(event_ids):
        val_end = len(event_ids) - 1
    split_map = {}
    for index, event_id in enumerate(event_ids):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "val"
        else:
            split = "test"
        split_map[event_id] = split
    return split_map


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(objects, edges, event_splits):
    summary = {
        "events": len(event_splits),
        "objects": len(objects),
        "edges": len(edges),
        "splits": {name: list(event_splits.values()).count(name) for name in ("train", "val", "test")},
        "relations": {},
        "labels": {},
    }
    for edge in edges:
        relation = edge["relation"]
        summary["relations"][relation] = summary["relations"].get(relation, 0) + 1
    for label in ("label_survive", "label_decay", "label_split", "label_merge", "label_initiation"):
        summary["labels"][label] = int(sum(int(obj.get(label, 0)) for obj in objects))
    return summary


def main():
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    event_dirs = find_event_dirs(data_root)
    if args.max_events > 0:
        event_dirs = event_dirs[: args.max_events]
    if not event_dirs:
        raise ValueError("No event directories with PNG frames found under {}".format(data_root))

    all_objects = []
    all_edges = []
    next_object_id = 0
    for event_index, event_dir in enumerate(event_dirs, 1):
        objects, edges, next_object_id = process_event(event_dir, data_root, args, next_object_id)
        all_objects.extend(objects)
        all_edges.extend(edges)
        print(
            "event {}/{} {} objects {} edges {}".format(
                event_index, len(event_dirs), relative_event_id(event_dir, data_root), len(objects), len(edges)
            ),
            flush=True,
        )

    event_ids = [relative_event_id(event_dir, data_root) for event_dir in event_dirs]
    event_splits = split_events(event_ids, args.train_ratio, args.val_ratio)
    for obj in all_objects:
        obj["split"] = event_splits[obj["event_id"]]
    for edge in all_edges:
        edge["split"] = event_splits[edge["event_id"]]

    object_fields = OBJECT_FIELDS + ["split"]
    edge_fields = EDGE_FIELDS + ["split"]
    write_csv(output_dir / "objects.csv", all_objects, object_fields)
    write_csv(output_dir / "edges.csv", all_edges, edge_fields)
    write_csv(
        output_dir / "event_splits.csv",
        [{"event_id": event_id, "split": split} for event_id, split in event_splits.items()],
        ["event_id", "split"],
    )
    summary = summarize(all_objects, all_edges, event_splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
