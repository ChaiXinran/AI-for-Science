#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_v3_object_consistency_tendency_facl}"
BASELINE_OBJECT_RUN_ROOT="${BASELINE_OBJECT_RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_v3_object}"
LEGACY_RUN_ROOT="${LEGACY_RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PWV_TENDENCY_WINDOWS="${PWV_TENDENCY_WINDOWS:-30,60}"
PWV_TENDENCY_MODE="${PWV_TENDENCY_MODE:-slope}"
FORECAST_LOSS="${FORECAST_LOSS:-facl}"
FACL_ALPHA="${FACL_ALPHA:-0.1}"
FACL_REDUCTION="${FACL_REDUCTION:-official}"

export DATA_ROOT RUN_ROOT BASELINE_OBJECT_RUN_ROOT LEGACY_RUN_ROOT
export DEVICE BATCH_SIZE TEST_BATCH_SIZE EPOCHS NUM_WORKERS
export PWV_TENDENCY_WINDOWS PWV_TENDENCY_MODE FORECAST_LOSS FACL_ALPHA FACL_REDUCTION

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
    --new_label "PWV V3 Object Consistency Tendency FACL" \
    --output_dir "${RUN_ROOT}/reports/v3_object_vs_consistency_tendency_facl" \
    2>&1 | tee "${RUN_ROOT}/logs/report_v3_object_vs_consistency_tendency_facl.log"
