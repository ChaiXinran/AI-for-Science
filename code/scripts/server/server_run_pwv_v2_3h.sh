#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
source "${SCRIPT_DIR}/server_defaults.sh"
init_server_defaults "pwv_v2_3h"
enter_code_dir "${SCRIPT_DIR}"

bash "${SCRIPT_DIR}/server_train_pwv_v2_3h.sh"
bash "${SCRIPT_DIR}/server_test_pwv_v2_3h.sh"
bash "${SCRIPT_DIR}/server_report_pwv_v2_3h.sh"

echo "V2 run root: ${RUN_ROOT}"
