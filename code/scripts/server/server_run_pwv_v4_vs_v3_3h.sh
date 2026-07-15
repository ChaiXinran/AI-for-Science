#!/usr/bin/env bash
set -euo pipefail

LEAD_TIME_EMBED_DIM="${LEAD_TIME_EMBED_DIM:-16}"
PWV_ATTN_DIM="${PWV_ATTN_DIM:-64}"
PWV_ATTN_HEADS="${PWV_ATTN_HEADS:-4}"
PWV_ATTN_DOWNSAMPLE="${PWV_ATTN_DOWNSAMPLE:-4}"
PWV_ATTN_SOURCE_SCALE="${PWV_ATTN_SOURCE_SCALE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
source "${SCRIPT_DIR}/server_defaults.sh"
init_server_defaults "pwv_v4_3h"
enter_code_dir "${SCRIPT_DIR}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-${RUNS_ROOT}/pwv_v3_3h}"
BASE_METRICS="${BASE_RUN_ROOT}/results/pwv_v3_3h/metrics.json"

if [[ ! -f "${BASE_METRICS}" ]]; then
    echo "Missing baseline V3 metrics: ${BASE_METRICS}" >&2
    echo "Run server_run_pwv_v3_3h.sh first, or set BASE_RUN_ROOT." >&2
    exit 1
fi

export LEAD_TIME_EMBED_DIM PWV_ATTN_DIM PWV_ATTN_HEADS PWV_ATTN_DOWNSAMPLE PWV_ATTN_SOURCE_SCALE

(
    cd "${CODE_DIR}"
    bash "${SCRIPT_DIR}/server_train_pwv_v4_3h.sh"
    bash "${SCRIPT_DIR}/server_test_pwv_v4_3h.sh"
    run_python_module nowcasting.cli.reports.pwv_model_compare \
        --baseline_run_root "${BASE_RUN_ROOT}" \
        --baseline_result_dir "pwv_v3_3h" \
        --baseline_label "PWV V3" \
        --new_run_root "${RUN_ROOT}" \
        --new_result_dir "pwv_v4_3h" \
        --new_label "PWV V4 Attention" \
        --output_dir "${RUN_ROOT}/reports/pwv_v4_vs_v3"
) 2>&1 | tee "${RUN_ROOT}/logs/run_pwv_v4_vs_v3_3h.log"

echo "V4 run root: ${RUN_ROOT}"
echo "V4 vs V3 report: ${RUN_ROOT}/reports/pwv_v4_vs_v3"
