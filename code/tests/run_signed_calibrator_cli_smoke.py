"""End-to-end CLI smoke test for the signed PWV calibrator."""

import json
import math
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
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "code")
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def write_dataset(root):
    rain_root = root / "rain"
    pwv_root = root / "pwv"
    splits = {"train": "20250701", "val": "20250702", "test": "20250703"}
    for split_index, day_token in enumerate(splits.values()):
        rain_day = rain_root / day_token
        pwv_day = pwv_root / day_token
        rain_day.mkdir(parents=True)
        pwv_day.mkdir(parents=True)
        start = datetime.strptime(day_token, "%Y%m%d")
        y, x = np.mgrid[:32, :32]
        for frame_index in range(5):
            stamp = start + timedelta(minutes=6 * frame_index)
            name = stamp.strftime("%Y-%m-%d-%H-%M-%S.png")
            rain_value = np.clip(
                250
                - 10 * frame_index
                - ((x - 9 - frame_index) ** 2 + (y - 16) ** 2),
                0,
                255,
            )
            pwv_value = np.clip(
                170 - split_index * 5 - frame_index * 3 - x // 4 - y // 4,
                0,
                255,
            )
            Image.fromarray(rain_value.astype(np.uint8)).save(rain_day / name)
            Image.fromarray(pwv_value.astype(np.uint8)).save(pwv_day / name)
    manifest = root / "split_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split_sha256": "synthetic-smoke",
                "splits": {name: [day] for name, day in splits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    climatology = root / "pwv_train_climatology.npz"
    np.savez_compressed(
        climatology,
        mean=np.full((32, 32), 30.0, dtype=np.float32),
        std=np.full((32, 32), 5.0, dtype=np.float32),
    )
    return rain_root, pwv_root, manifest, climatology


def main():
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory(prefix="pwv_signed_cli_") as temp:
        root = Path(temp)
        rain_root, pwv_root, manifest, climatology = write_dataset(root)
        output = root / "output"
        radar_checkpoint = output / "radar" / "best_state_dict.ckpt"
        signed_checkpoint = output / "signed" / "best_state_dict.ckpt"
        shape = [
            "--input_length",
            "2",
            "--total_length",
            "4",
            "--img_height",
            "32",
            "--img_width",
            "32",
        ]
        common_data = [
            "--data_root",
            rain_root,
            "--split_manifest",
            manifest,
            "--require_contiguous",
        ]
        run(
            "code/train/radar.py",
            *common_data,
            "--save_dir",
            output / "radar",
            "--readme_ckpt",
            output / "radar.ckpt",
            "--device",
            device,
            "--batch_size",
            "2",
            "--epochs",
            "1",
            "--num_workers",
            "0",
            "--max_train_samples",
            "2",
            "--max_val_samples",
            "2",
            "--ngf",
            "4",
            "--lead_time_embed_dim",
            "4",
            "--disc_channels",
            "4",
            "--intensity_scale",
            "35",
            "--log_interval",
            "1",
            *shape,
        )
        run(
            "code/train/pwv_signed.py",
            *common_data,
            "--pwv_root",
            pwv_root,
            "--strict_pwv",
            "--save_dir",
            output / "signed",
            "--readme_ckpt",
            output / "signed.ckpt",
            "--init_radar_checkpoint",
            radar_checkpoint,
            "--freeze_radar_backbone",
            "--pwv_climatology_path",
            climatology,
            "--device",
            device,
            "--batch_size",
            "2",
            "--epochs",
            "1",
            "--num_workers",
            "0",
            "--max_train_samples",
            "2",
            "--max_val_samples",
            "2",
            "--ngf",
            "4",
            "--lead_time_embed_dim",
            "4",
            "--evo_base_channels",
            "32",
            "--pwv_base_channels",
            "4",
            "--fusion_channels",
            "4",
            "--intensity_scale",
            "35",
            "--pwv_intensity_scale",
            "80",
            "--pwv_invert",
            "--calibration_thresholds",
            "10,20",
            "--log_interval",
            "1",
            *shape,
        )
        test_common = [
            *common_data,
            "--pwv_root",
            pwv_root,
            "--strict_pwv",
            "--checkpoint",
            signed_checkpoint,
            "--split",
            "test",
            "--model_name",
            "PWVSignedCalibratorNowcastNet",
            "--pwv_climatology_path",
            climatology,
            "--device",
            device,
            "--batch_size",
            "1",
            "--num_workers",
            "0",
            "--max_samples",
            "1",
            "--num_save_samples",
            "0",
            "--ngf",
            "4",
            "--lead_time_embed_dim",
            "4",
            "--evo_base_channels",
            "32",
            "--pwv_base_channels",
            "4",
            "--fusion_channels",
            "4",
            "--intensity_scale",
            "35",
            "--pwv_intensity_scale",
            "80",
            "--pwv_invert",
            "--metric_thresholds",
            "10,20",
            "--neighborhood_metric_thresholds",
            "10,20",
            "--object_thresholds",
            "10,20",
            "--cra_thresholds",
            "10,20",
            "--psd_lead_minutes",
            "6,12",
            "--cra_lead_minutes",
            "6,12",
            "--horizon_bins",
            "0-0.2",
            "--deterministic_noise",
            *shape,
        ]
        for control in ("real", "zero", "level_only", "spatial_shift"):
            run(
                "code/test/pwv.py",
                *test_common,
                "--pwv_control",
                control,
                "--output_dir",
                output / control,
            )
        real_metrics = json.loads(
            (output / "real" / "metrics.json").read_text(encoding="utf-8")
        )
        zero_metrics = json.loads(
            (output / "zero" / "metrics.json").read_text(encoding="utf-8")
        )
        if real_metrics["samples"] != 1 or zero_metrics["samples"] != 1:
            raise RuntimeError("Unexpected smoke-test sample count.")
        for metric in ("mae", "mse", "rmse", "bias"):
            if not math.isfinite(float(zero_metrics["model"][metric])):
                raise RuntimeError("Non-finite null metric: {}".format(metric))
        print("SIGNED_CLI_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
