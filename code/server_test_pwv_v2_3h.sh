#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results"

python -u test_pwv_coupled_v2.py \
    --data_root "${DATA_ROOT}/RADAR_2025_S" \
    --pwv_root "${DATA_ROOT}/PWV_2025_S" \
    --checkpoint "${RUN_ROOT}/checkpoints/pwv_v2_3h_model.ckpt" \
    --output_dir "${RUN_ROOT}/results/pwv_v2_3h" \
    --device "${DEVICE}" \
    --split test \
    --input_length 9 \
    --total_length 39 \
    --img_height 96 \
    --img_width 96 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --stride 1 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --max_samples 0 \
    --num_save_samples 24 \
    --metric_thresholds "1,5,10,20,40" \
    --frame_minutes 6 \
    --horizon_bins "0-1,1-2,2-3" \
    2>&1 | tee "${RUN_ROOT}/logs/test_pwv_v2_3h.log"
