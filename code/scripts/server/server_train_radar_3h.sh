#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/runtime.sh"
source "${SCRIPT_DIR}/server_defaults.sh"
source "${SCRIPT_DIR}/server_env.sh"
init_server_defaults "radar_3h"
enter_code_dir "${SCRIPT_DIR}"
RADAR_ROOT="${RADAR_ROOT:-$(resolve_dataset_dir "${DATA_ROOT}" "RADAR_2025_S" "*RADAR*")}"
RAIN_ROOT="${RAIN_ROOT:-$(try_resolve_dataset_dir "${DATA_ROOT}" "RAIN_2025_S" "*RAIN*" || true)}"
PRECIP_ROOT="${PRECIP_ROOT:-${RAIN_ROOT:-${RADAR_ROOT}}}"
PRECIP_SCALE="${PRECIP_SCALE:-35}"

print_dataset_dir "RADAR_ROOT" "${RADAR_ROOT}"
print_dataset_dir "PRECIP_ROOT" "${PRECIP_ROOT}"

run_python_module nowcasting.cli.custom.train_adversarial \
    --data_root "${PRECIP_ROOT}" \
    --save_dir "${RUN_ROOT}/checkpoints/radar_3h" \
    --readme_ckpt "${RUN_ROOT}/checkpoints/radar_3h_model.ckpt" \
    --device "${DEVICE}" \
    --input_length 9 \
    --total_length 39 \
    --img_height 96 \
    --img_width 96 \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --stride 1 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --max_train_samples 0 \
    --max_val_samples 0 \
    --intensity_scale "${PRECIP_SCALE}" \
    --pixel_min 0 \
    --pixel_max 255 \
    --disc_channels 32 \
    --lead_time_embed_dim 16 \
    --lambda_forecast 1.0 \
    --lambda_evolution 0.5 \
    --lambda_advected 0.25 \
    --lambda_motion 0.02 \
    --lambda_pool 0.2 \
    --lambda_adv 0.01 \
    --grad_clip 1.0 \
    --log_interval 100 \
    --amp \
    2>&1 | tee "${RUN_ROOT}/logs/train_radar_3h.log"
