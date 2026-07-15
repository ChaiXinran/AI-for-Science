#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical_pwv_v3_tendency}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEAD_TIME_EMBED_DIM="${LEAD_TIME_EMBED_DIM:-16}"
PWV_TENDENCY_WINDOWS="${PWV_TENDENCY_WINDOWS:-18,36,48}"
PWV_TENDENCY_MODE="${PWV_TENDENCY_MODE:-both}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_METRICS="${BASE_RUN_ROOT}/results/pwv_v3_3h/metrics.json"

if [[ ! -f "${BASE_METRICS}" ]]; then
    echo "Missing baseline V3 metrics: ${BASE_METRICS}" >&2
    echo "Run the original V3 experiment first, or set BASE_RUN_ROOT to a folder containing results/pwv_v3_3h/metrics.json." >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" "${RUN_ROOT}/reports"

export DATA_ROOT RUN_ROOT DEVICE BATCH_SIZE TEST_BATCH_SIZE EPOCHS NUM_WORKERS
export LEAD_TIME_EMBED_DIM PWV_TENDENCY_WINDOWS PWV_TENDENCY_MODE

(
    cd "${SCRIPT_DIR}"
    bash ./server_train_pwv_v3_3h.sh
    bash ./server_test_pwv_v3_3h.sh
    python -u make_pwv_model_compare_report.py \
        --baseline_run_root "${BASE_RUN_ROOT}" \
        --baseline_result_dir "pwv_v3_3h" \
        --baseline_label "PWV V3" \
        --new_run_root "${RUN_ROOT}" \
        --new_result_dir "pwv_v3_3h" \
        --new_label "PWV V3 Tendency" \
        --output_dir "${RUN_ROOT}/reports/pwv_v3_tendency_vs_v3"
) 2>&1 | tee "${RUN_ROOT}/logs/run_pwv_v3_tendency_3h.log"

echo "V3 tendency run root: ${RUN_ROOT}"
echo "V3 tendency vs V3 report: ${RUN_ROOT}/reports/pwv_v3_tendency_vs_v3"
