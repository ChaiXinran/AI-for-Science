#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
source "${SCRIPT_DIR}/server_defaults.sh"
init_server_defaults "pwv_v4_tendency_3h"

export PWV_TENDENCY_WINDOWS="${PWV_TENDENCY_WINDOWS:-30,60}"
export PWV_TENDENCY_MODE="${PWV_TENDENCY_MODE:-slope}"
export FRAME_MINUTES="${FRAME_MINUTES:-6}"

bash "${SCRIPT_DIR}/server_train_pwv_v4_3h.sh"
