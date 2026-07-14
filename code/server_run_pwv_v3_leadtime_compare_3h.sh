#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical}"
NEW_RUN_ROOT="${NEW_RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical_pwv_v3_leadtime}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEAD_TIME_EMBED_DIM="${LEAD_TIME_EMBED_DIM:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_METRICS="${BASE_RUN_ROOT}/results/pwv_v3_3h/metrics.json"

if [[ ! -f "${BASE_METRICS}" ]]; then
    echo "Missing original V3 metrics: ${BASE_METRICS}" >&2
    echo "Set BASE_RUN_ROOT to the run folder that contains results/pwv_v3_3h/metrics.json." >&2
    exit 1
fi

mkdir -p "${NEW_RUN_ROOT}/logs" "${NEW_RUN_ROOT}/checkpoints" "${NEW_RUN_ROOT}/results" "${NEW_RUN_ROOT}/reports"

export DATA_ROOT DEVICE BATCH_SIZE TEST_BATCH_SIZE EPOCHS NUM_WORKERS
export RUN_ROOT="${NEW_RUN_ROOT}"
export LEAD_TIME_EMBED_DIM

(
    cd "${SCRIPT_DIR}"
    bash ./server_train_pwv_v3_3h.sh
    bash ./server_test_pwv_v3_3h.sh
    python -u make_pwv_v3_leadtime_compare_report.py \
        --baseline_run_root "${BASE_RUN_ROOT}" \
        --new_run_root "${NEW_RUN_ROOT}" \
        --baseline_label "Original PWV V3" \
        --new_label "Lead-time PWV V3" \
        --output_dir "${NEW_RUN_ROOT}/reports/pwv_v3_leadtime_compare"
) 2>&1 | tee "${NEW_RUN_ROOT}/logs/run_pwv_v3_leadtime_compare_3h.log"

echo "New V3 run root: ${NEW_RUN_ROOT}"
echo "Comparison report: ${NEW_RUN_ROOT}/reports/pwv_v3_leadtime_compare"
