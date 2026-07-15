#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
enter_code_dir "${SCRIPT_DIR}"

run_python_module nowcasting.cli.custom.test \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --checkpoint ../checkpoints/custom_nowcastnet/best_state_dict.ckpt \
    --output_dir ../results/custom_test \
    --device cuda:0 \
    --split test \
    --img_height 96 \
    --img_width 96 \
    --batch_size 1 \
    --num_workers 2 \
    --max_samples 20 \
    --num_save_samples 10 \
    --intensity_scale 128 \
    --pixel_min 0 \
    --pixel_max 255 \
    --metric_thresholds 1,5,10,20,40
