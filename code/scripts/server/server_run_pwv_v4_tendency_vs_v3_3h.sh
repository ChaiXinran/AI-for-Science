#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
source "${SCRIPT_DIR}/server_defaults.sh"
init_server_defaults "pwv_v4_tendency_3h"
enter_code_dir "${SCRIPT_DIR}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-${RUNS_ROOT}/pwv_v3_3h}"
BASE_METRICS="${BASE_RUN_ROOT}/results/pwv_v3_3h/metrics.json"

if [[ ! -f "${BASE_METRICS}" ]]; then
    echo "Missing baseline V3 metrics: ${BASE_METRICS}" >&2
    echo "Run server_run_pwv_v3_3h.sh first, or set BASE_RUN_ROOT." >&2
    exit 1
fi

export PWV_TENDENCY_WINDOWS="${PWV_TENDENCY_WINDOWS:-30,60}"
export PWV_TENDENCY_MODE="${PWV_TENDENCY_MODE:-slope}"
export FRAME_MINUTES="${FRAME_MINUTES:-6}"

(
    cd "${CODE_DIR}"
    bash "${SCRIPT_DIR}/server_train_pwv_v4_tendency_3h.sh"
    bash "${SCRIPT_DIR}/server_test_pwv_v4_tendency_3h.sh"
    run_python_module nowcasting.cli.reports.pwv_model_compare \
        --baseline_run_root "${BASE_RUN_ROOT}" \
        --baseline_result_dir "pwv_v3_3h" \
        --baseline_label "PWV V3" \
        --new_run_root "${RUN_ROOT}" \
        --new_result_dir "pwv_v4_3h" \
        --new_label "PWV V4 Tendency" \
        --output_dir "${RUN_ROOT}/reports/pwv_v4_tendency_vs_v3"
) 2>&1 | tee "${RUN_ROOT}/logs/run_pwv_v4_tendency_vs_v3_3h.log"

echo "V4 tendency run root: ${RUN_ROOT}"
echo "V4 tendency vs V3 report: ${RUN_ROOT}/reports/pwv_v4_tendency_vs_v3"
