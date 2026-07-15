#!/usr/bin/env bash
set -euo pipefail

LEAD_TIME_EMBED_DIM="${LEAD_TIME_EMBED_DIM:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
source "${SCRIPT_DIR}/server_defaults.sh"
init_server_defaults "pwv_v3_leadtime_3h"
enter_code_dir "${SCRIPT_DIR}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-${RUNS_ROOT}/pwv_v3_3h}"
NEW_RUN_ROOT="${NEW_RUN_ROOT:-${RUN_ROOT}}"
BASE_METRICS="${BASE_RUN_ROOT}/results/pwv_v3_3h/metrics.json"

if [[ ! -f "${BASE_METRICS}" ]]; then
    echo "Missing original V3 metrics: ${BASE_METRICS}" >&2
    echo "Set BASE_RUN_ROOT to the run folder that contains results/pwv_v3_3h/metrics.json." >&2
    exit 1
fi

export RUN_ROOT="${NEW_RUN_ROOT}"
export LEAD_TIME_EMBED_DIM

(
    cd "${CODE_DIR}"
    bash "${SCRIPT_DIR}/server_train_pwv_v3_3h.sh"
    bash "${SCRIPT_DIR}/server_test_pwv_v3_3h.sh"
    run_python_module nowcasting.cli.reports.pwv_v3_leadtime_compare \
        --baseline_run_root "${BASE_RUN_ROOT}" \
        --new_run_root "${NEW_RUN_ROOT}" \
        --baseline_label "Original PWV V3" \
        --new_label "Lead-time PWV V3" \
        --output_dir "${NEW_RUN_ROOT}/reports/pwv_v3_leadtime_compare"
) 2>&1 | tee "${NEW_RUN_ROOT}/logs/run_pwv_v3_leadtime_compare_3h.log"

echo "New V3 run root: ${NEW_RUN_ROOT}"
echo "Comparison report: ${NEW_RUN_ROOT}/reports/pwv_v3_leadtime_compare"
