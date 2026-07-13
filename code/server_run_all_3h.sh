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

bash ./server_train_radar_3h.sh
bash ./server_test_radar_3h.sh
bash ./server_train_pwv_v2_3h.sh
bash ./server_test_pwv_v2_3h.sh
bash ./server_train_pwv_v3_3h.sh
bash ./server_test_pwv_v3_3h.sh

python -u make_server_3h_report.py \
    --run_root "${RUN_ROOT}" \
    --output_dir "${RUN_ROOT}/reports/comparison_3h" \
    2>&1 | tee "${RUN_ROOT}/logs/report_3h.log"
