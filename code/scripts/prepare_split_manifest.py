"""Create a deterministic day-blocked split manifest for PNG sequences.

Day directories are the minimum safe grouping unit available from the current
repository layout. For publication, review the generated event list and merge
adjacent days belonging to the same storm before locking the manifest.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def discover_events(data_root):
    events = []
    for path in sorted(p for p in data_root.rglob("*") if p.is_dir()):
        files = sorted(path.glob("*.png"))
        if not files:
            continue
        events.append(
            {
                "path": path.relative_to(data_root).as_posix(),
                "frames": len(files),
                "first_file": files[0].name,
                "last_file": files[-1].name,
            }
        )
    return events


def split_counts(n, train_ratio, val_ratio):
    if n < 3:
        raise ValueError("At least 3 day/event directories are required for train/val/test.")
    train_n = max(1, int(n * train_ratio))
    val_n = max(1, int(n * val_ratio))
    if train_n + val_n >= n:
        train_n = max(1, n - 2)
        val_n = 1
    return train_n, val_n


def _normalise_boundary(value):
    return "".join(character for character in value if character.isdigit())


def split_at_explicit_boundaries(events, train_end, val_end):
    """Split after two inclusive YYYYMMDD boundaries."""
    train_token = _normalise_boundary(train_end)
    val_token = _normalise_boundary(val_end)
    if len(train_token) != 8 or len(val_token) != 8:
        raise ValueError("Explicit boundaries must be YYYYMMDD or YYYY-MM-DD.")
    event_tokens = [_normalise_boundary(Path(item["path"]).name) for item in events]
    try:
        train_end_index = event_tokens.index(train_token)
        val_end_index = event_tokens.index(val_token)
    except ValueError as error:
        raise ValueError("Explicit boundary date is not present in the discovered events.") from error
    if not 0 <= train_end_index < val_end_index < len(events) - 1:
        raise ValueError("Boundaries must leave non-empty chronological train, val, and test splits.")
    names = [item["path"] for item in events]
    return {
        "train": names[: train_end_index + 1],
        "val": names[train_end_index + 1 : val_end_index + 1],
        "test": names[val_end_index + 1 :],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pwv_root", default="")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--train_end", default="", help="Inclusive YYYYMMDD boundary; requires --val_end.")
    parser.add_argument("--val_end", default="", help="Inclusive YYYYMMDD boundary; requires --train_end.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    events = discover_events(root)
    missing_pwv = []
    if args.pwv_root:
        pwv_root = Path(args.pwv_root).resolve()
        for item in events:
            event_dir = root / item["path"]
            for frame in event_dir.glob("*.png"):
                relative = frame.relative_to(root)
                if not (pwv_root / relative).exists():
                    missing_pwv.append(relative.as_posix())
        if missing_pwv:
            raise ValueError(
                "PWV pairing audit failed: {} frames missing; first: {}".format(
                    len(missing_pwv), missing_pwv[0]
                )
            )
    if bool(args.train_end) != bool(args.val_end):
        raise ValueError("--train_end and --val_end must be supplied together.")
    if args.train_end:
        splits = split_at_explicit_boundaries(events, args.train_end, args.val_end)
        boundary_mode = "explicit_meteorological_review"
        warning = "Locked to explicitly reviewed inclusive date boundaries."
    else:
        train_n, val_n = split_counts(len(events), args.train_ratio, args.val_ratio)
        # Chronological blocking is intentional: it prevents future storms from
        # leaking into training and makes the split stable across machines.
        names = [item["path"] for item in events]
        splits = {
            "train": names[:train_n],
            "val": names[train_n:train_n + val_n],
            "test": names[train_n + val_n:],
        }
        boundary_mode = "ratio_draft_requires_review"
        warning = "Review split boundaries and move adjacent same-storm days to the same split before locking."
    digest_payload = json.dumps(splits, sort_keys=True).encode("utf-8")
    manifest = {
        "protocol_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_root_name": root.name,
        "grouping": "day_directory_chronological_block",
        "seed_reserved_for_training": args.seed,
        "requested_train_ratio": args.train_ratio,
        "requested_val_ratio": args.val_ratio,
        "actual_ratios": {
            name: len(items) / len(events) for name, items in splits.items()
        },
        "boundary_mode": boundary_mode,
        "train_end": args.train_end or splits["train"][-1],
        "val_end": args.val_end or splits["val"][-1],
        "split_sha256": hashlib.sha256(digest_payload).hexdigest(),
        "warning": warning,
        "pwv_pairing_checked": bool(args.pwv_root),
        "missing_pwv_frames": len(missing_pwv),
        "events": events,
        "splits": splits,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote {} (train={} val={} test={} day_directories={})".format(
        output, len(splits["train"]), len(splits["val"]), len(splits["test"]), len(events)
    ))


if __name__ == "__main__":
    main()
