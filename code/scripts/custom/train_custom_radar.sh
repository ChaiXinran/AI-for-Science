#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
enter_code_dir "${SCRIPT_DIR}"

run_python_module nowcasting.cli.custom.train \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --save_dir ../checkpoints/custom_nowcastnet \
    --device cuda:0 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 4 \
    --epochs 20 \
    --lr 0.0001 \
    --num_workers 2 \
    --stride 1 \
    --intensity_scale 128 \
    --pixel_min 0 \
    --pixel_max 255 \
    --log_interval 20
