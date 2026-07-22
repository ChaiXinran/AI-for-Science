"""Run the real train/test CLIs on a tiny temporary paired dataset."""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    command = [sys.executable, "-u", *map(str, args)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(), check=True)


def write_dataset(root):
    radar_root = root / "radar"
    pwv_root = root / "pwv"
    split_days = {"train": "2025-07-01", "val": "2025-07-02", "test": "2025-07-03"}
    for split_index, day in enumerate(split_days.values()):
        radar_day = radar_root / day
        pwv_day = pwv_root / day
        radar_day.mkdir(parents=True)
        pwv_day.mkdir(parents=True)
        start = datetime.strptime(day, "%Y-%m-%d")
        y, x = np.mgrid[:32, :32]
        for frame_index in range(5):
            stamp = start + timedelta(minutes=6 * frame_index)
            name = stamp.strftime("%Y-%m-%d-%H-%M-%S.png")
            radar = np.clip(240 - 8 * frame_index - ((x - 8 - frame_index) ** 2 + (y - 16) ** 2), 0, 255)
            pwv = np.clip(80 + split_index * 10 + frame_index * 8 + x + y, 0, 255)
            Image.fromarray(radar.astype("uint8")).save(radar_day / name)
            Image.fromarray(pwv.astype("uint8")).save(pwv_day / name)
    manifest = root / "split_manifest.json"
    manifest.write_text(
        json.dumps({"splits": {name: [day] for name, day in split_days.items()}}, indent=2),
        encoding="utf-8",
    )
    return radar_root, pwv_root, manifest


def main():
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("This integration smoke test requires a CUDA-enabled PyTorch.")
    with tempfile.TemporaryDirectory(prefix="nowcast_birth_growth_cli_") as temp:
        root = Path(temp)
        radar_root, pwv_root, manifest = write_dataset(root)
        out = root / "out"
        radar_ckpt = out / "radar" / "best_state_dict.ckpt"
        pwv_ckpt = out / "pwv" / "best_state_dict.ckpt"
        shape = ["--input_length", "2", "--total_length", "4", "--img_height", "32", "--img_width", "32"]
        data = ["--data_root", radar_root, "--split_manifest", manifest, "--require_contiguous"]
        train = [
            "--device", "cuda:0", "--batch_size", "2", "--epochs", "1", "--num_workers", "0",
            "--max_train_samples", "2", "--max_val_samples", "2", "--ngf", "4",
            "--lead_time_embed_dim", "4", "--disc_channels", "4", "--intensity_scale", "35",
            "--log_interval", "1",
        ]
        test = [
            "--device", "cuda:0", "--batch_size", "1", "--num_workers", "0", "--max_samples", "1",
            "--num_save_samples", "0", "--intensity_scale", "35", "--metric_thresholds", "2,10",
            "--neighborhood_metric_thresholds", "10", "--object_thresholds", "10",
            "--cra_thresholds", "10", "--psd_lead_minutes", "6,12", "--cra_lead_minutes", "6,12",
        ]

        run("code/train/radar.py", *data, "--save_dir", out / "radar", "--readme_ckpt", out / "radar.ckpt", *shape, *train)
        run(
            "code/test/radar.py", *data, "--checkpoint", radar_ckpt, "--output_dir", out / "radar_test",
            "--split", "test", *shape, "--ngf", "4", "--lead_time_embed_dim", "4", *test,
        )
        run(
            "code/train/pwv.py", *data, "--pwv_root", pwv_root, "--strict_pwv",
            "--save_dir", out / "pwv", "--readme_ckpt", out / "pwv.ckpt",
            "--init_radar_checkpoint", radar_ckpt, "--freeze_radar_backbone",
            "--model_name", "PWVBirthGrowthNowcastNet", *shape, *train,
            "--evo_base_channels", "32", "--pwv_base_channels", "4", "--fusion_channels", "4",
            "--pwv_intensity_scale", "80", "--pwv_invert", "--pwv_tendency_windows", "6,12",
        )
        run(
            "code/test/pwv.py", *data, "--pwv_root", pwv_root, "--strict_pwv",
            "--checkpoint", pwv_ckpt, "--output_dir", out / "pwv_test", "--split", "test",
            "--model_name", "PWVBirthGrowthNowcastNet", *shape,
            "--ngf", "4", "--lead_time_embed_dim", "4", "--evo_base_channels", "32",
            "--pwv_base_channels", "4", "--fusion_channels", "4", "--pwv_intensity_scale", "80",
            "--pwv_invert", "--pwv_tendency_windows", "6,12", *test,
        )
        for result in (out / "radar_test" / "metrics.json", out / "pwv_test" / "metrics.json"):
            if not result.exists():
                raise RuntimeError("Missing CLI smoke result: {}".format(result))
        print("CLI_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
