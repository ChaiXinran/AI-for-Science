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

export LEAD_TIME_EMBED_DIM PWV_ATTN_DIM PWV_ATTN_HEADS PWV_ATTN_DOWNSAMPLE PWV_ATTN_SOURCE_SCALE

bash "${SCRIPT_DIR}/server_train_pwv_v4_3h.sh"
bash "${SCRIPT_DIR}/server_test_pwv_v4_3h.sh"
bash "${SCRIPT_DIR}/server_report_pwv_v4_3h.sh"

echo "V4 run root: ${RUN_ROOT}"
