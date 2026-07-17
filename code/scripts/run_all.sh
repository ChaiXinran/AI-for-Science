#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export DATA_ROOT RUN_ROOT DEVICE BATCH_SIZE TEST_BATCH_SIZE EPOCHS NUM_WORKERS

# Radar-only baseline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/train_radar.sh"
bash "${SCRIPT_DIR}/test_radar.sh"

# PWV-coupled model
python -u train/pwv.py \
    --data_root "${DATA_ROOT}/RADAR_2025_S" \
    --pwv_root "${DATA_ROOT}/PWV_2025_S" \
    --precip_root "${DATA_ROOT}/RAIN_2025_S" \
    --save_dir "${RUN_ROOT}/checkpoints/pwv_3h" \
    --readme_ckpt "${RUN_ROOT}/checkpoints/pwv_3h_model.ckpt" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --total_length 39 \
    --intensity_scale 35 \
    --pwv_intensity_scale 80 --pwv_invert \
    --metric_thresholds 0.5,2,5,10,30 \
    --neighborhood_metric_thresholds 16,32 --neighborhood_size 5 \
    2>&1 | tee "${RUN_ROOT}/logs/train_pwv_3h.log"

python -u test/pwv.py \
    --data_root "${DATA_ROOT}/RAIN_2025_S" \
    --pwv_root "${DATA_ROOT}/PWV_2025_S" \
    --checkpoint "${RUN_ROOT}/checkpoints/pwv_3h_model.ckpt" \
    --output_dir "${RUN_ROOT}/results/pwv_3h" \
    --device "${DEVICE}" \
    --batch_size "${TEST_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --total_length 39 \
    --intensity_scale 35 \
    --pwv_intensity_scale 80 --pwv_invert \
    --metric_thresholds 0.5,2,5,10,30 \
    --neighborhood_metric_thresholds 16,32 --neighborhood_size 5 \
    2>&1 | tee "${RUN_ROOT}/logs/test_pwv_3h.log"

# Report
python -u report/compare.py \
    --run_root "${RUN_ROOT}" \
    --output_dir "${RUN_ROOT}/reports/comparison_3h" \
    2>&1 | tee "${RUN_ROOT}/logs/report_3h.log"
