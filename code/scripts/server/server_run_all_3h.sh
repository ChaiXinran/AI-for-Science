#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
source "${SCRIPT_DIR}/server_defaults.sh"
init_server_defaults "all_3h_comparison"
enter_code_dir "${SCRIPT_DIR}"

bash "${SCRIPT_DIR}/server_train_radar_3h.sh"
bash "${SCRIPT_DIR}/server_test_radar_3h.sh"
bash "${SCRIPT_DIR}/server_train_pwv_v1_3h.sh"
bash "${SCRIPT_DIR}/server_test_pwv_v1_3h.sh"
bash "${SCRIPT_DIR}/server_train_pwv_v2_3h.sh"
bash "${SCRIPT_DIR}/server_test_pwv_v2_3h.sh"
bash "${SCRIPT_DIR}/server_train_pwv_v3_3h.sh"
bash "${SCRIPT_DIR}/server_test_pwv_v3_3h.sh"
bash "${SCRIPT_DIR}/server_train_pwv_v4_3h.sh"
bash "${SCRIPT_DIR}/server_test_pwv_v4_3h.sh"

run_server_report "comparison_3h"

echo "All-version run root: ${RUN_ROOT}"
