"""Audit rain-event prevalence for a frozen split without running a model."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def read_rain(path, intensity_scale):
    if cv2 is not None:
        pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise ValueError("Could not read {}".format(path))
        pixels = pixels.astype(np.float32)
    else:
        pixels = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return (255.0 - pixels) / 255.0 * intensity_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intensity_scale", type=float, default=35.0)
    parser.add_argument("--input_length", type=int, default=9)
    parser.add_argument("--total_length", type=int, default=39)
    parser.add_argument("--thresholds", default="2,10,20")
    args = parser.parse_args()

    root = Path(args.data_root)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    report = {
        "data_root": str(root),
        "split_sha256": manifest.get("split_sha256"),
        "intensity_scale": args.intensity_scale,
        "input_length": args.input_length,
        "total_length": args.total_length,
        "thresholds": thresholds,
        "splits": {},
    }

    for split, relative_days in manifest["splits"].items():
        pixel_counts = {threshold: 0 for threshold in thresholds}
        positive_frames = {threshold: 0 for threshold in thresholds}
        positive_windows = {threshold: 0 for threshold in thresholds}
        window_pixel_occurrences = {threshold: 0 for threshold in thresholds}
        transition_birth = 0
        transition_growth = 0
        transition_pixels = 0
        total_pixels = 0
        total_frames = 0
        total_windows = 0
        day_rows = []
        for relative_day in relative_days:
            files = sorted((root / relative_day).glob("*.png"))
            frames = np.stack([read_rain(path, args.intensity_scale) for path in files])
            frame_pixels = frames.shape[1] * frames.shape[2]
            total_frames += len(frames)
            total_pixels += frames.size
            day_row = {"day": relative_day, "frames": len(frames), "max_mm_h": float(frames.max())}
            for threshold in thresholds:
                per_frame = (frames >= threshold).reshape(len(frames), -1).sum(axis=1)
                pixel_counts[threshold] += int(per_frame.sum())
                positive_frames[threshold] += int((per_frame > 0).sum())
                day_row["pixels_ge_{:g}".format(threshold)] = int(per_frame.sum())
                day_row["frames_ge_{:g}".format(threshold)] = int((per_frame > 0).sum())
                window_count = max(0, len(frames) - args.total_length + 1)
                for start in range(window_count):
                    future_count = int(per_frame[start + args.input_length : start + args.total_length].sum())
                    window_pixel_occurrences[threshold] += future_count
                    positive_windows[threshold] += int(future_count > 0)
            window_count = max(0, len(frames) - args.total_length + 1)
            total_windows += window_count
            if len(frames) > 1:
                previous = frames[:-1]
                current = frames[1:]
                transition_birth += int(((previous < 2.0) & (current >= 10.0)).sum())
                transition_growth += int(((previous >= 2.0) & ((current - previous) > 5.0)).sum())
                transition_pixels += previous.size
            day_rows.append(day_row)

        split_report = {
            "days": len(relative_days),
            "frames": total_frames,
            "windows": total_windows,
            "unique_frame_pixel_count": total_pixels,
            "six_minute_transition_pixel_count": transition_pixels,
            "six_minute_birth_count": transition_birth,
            "six_minute_birth_rate": transition_birth / transition_pixels if transition_pixels else None,
            "six_minute_growth_count": transition_growth,
            "six_minute_growth_rate": transition_growth / transition_pixels if transition_pixels else None,
            "by_threshold": {},
        }
        for threshold in thresholds:
            key = "{:g}".format(threshold)
            split_report["by_threshold"][key] = {
                "unique_pixel_count": pixel_counts[threshold],
                "unique_pixel_rate": pixel_counts[threshold] / total_pixels if total_pixels else None,
                "positive_frames": positive_frames[threshold],
                "positive_frame_rate": positive_frames[threshold] / total_frames if total_frames else None,
                "positive_windows": positive_windows[threshold],
                "positive_window_rate": positive_windows[threshold] / total_windows if total_windows else None,
                "target_pixel_occurrences_across_windows": window_pixel_occurrences[threshold],
            }
        split_report["top_days_by_10mm_pixels"] = sorted(
            day_rows, key=lambda row: row.get("pixels_ge_10", 0), reverse=True
        )[:10]
        report["splits"][split] = split_report

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
