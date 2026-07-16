#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_v3_object_consistency}"
BASELINE_OBJECT_RUN_ROOT="${BASELINE_OBJECT_RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_v3_object}"
LEGACY_RUN_ROOT="${LEGACY_RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export DATA_ROOT RUN_ROOT BASELINE_OBJECT_RUN_ROOT LEGACY_RUN_ROOT DEVICE BATCH_SIZE TEST_BATCH_SIZE EPOCHS NUM_WORKERS

bash ./server_train_pwv_v3_object_consistency_3h.sh
bash ./server_test_pwv_v3_object_consistency_3h.sh

COMPARE_BASELINE_RUN_ROOT="${COMPARE_BASELINE_RUN_ROOT:-${BASELINE_OBJECT_RUN_ROOT}}"
if [[ ! -f "${COMPARE_BASELINE_RUN_ROOT}/results/pwv_v3_object_3h/metrics.json" ]]; then
    COMPARE_BASELINE_RUN_ROOT="${LEGACY_RUN_ROOT}"
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/reports"
python -u make_pwv_model_compare_report.py \
    --baseline_run_root "${COMPARE_BASELINE_RUN_ROOT}" \
    --baseline_result_dir "pwv_v3_object_3h" \
    --baseline_label "PWV V3 Object" \
    --new_run_root "${RUN_ROOT}" \
    --new_result_dir "pwv_v3_object_consistency_3h" \
    --new_label "PWV V3 Object Consistency" \
    --output_dir "${RUN_ROOT}/reports/v3_object_vs_consistency" \
    2>&1 | tee "${RUN_ROOT}/logs/report_v3_object_vs_consistency.log"
