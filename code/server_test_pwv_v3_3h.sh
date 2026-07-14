#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/nowcastnet_runs/north_china_3h_physical}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/server_env.sh"
RADAR_ROOT="${RADAR_ROOT:-$(resolve_dataset_dir "${DATA_ROOT}" "RADAR_2025_S" "*RADAR*")}"
RAIN_ROOT="${RAIN_ROOT:-$(try_resolve_dataset_dir "${DATA_ROOT}" "RAIN_2025_S" "*RAIN*" || true)}"
PRECIP_ROOT="${PRECIP_ROOT:-${RAIN_ROOT:-${RADAR_ROOT}}}"
PWV_ROOT="${PWV_ROOT:-$(resolve_dataset_dir "${DATA_ROOT}" "PWV_2025_S" "*PWV*")}"
PRECIP_SCALE="${PRECIP_SCALE:-35}"
PWV_SCALE="${PWV_SCALE:-80}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
print_dataset_dir "RADAR_ROOT" "${RADAR_ROOT}"
print_dataset_dir "PRECIP_ROOT" "${PRECIP_ROOT}"
print_dataset_dir "PWV_ROOT" "${PWV_ROOT}"

python -u test_pwv_coupled_v2.py \
    --data_root "${PRECIP_ROOT}" \
    --pwv_root "${PWV_ROOT}" \
    --checkpoint "${RUN_ROOT}/checkpoints/pwv_v3_3h_model.ckpt" \
    --output_dir "${RUN_ROOT}/results/pwv_v3_3h" \
    --device "${DEVICE}" \
    --model_name PWVCoupledNowcastNetV3 \
    --split test \
    --input_length 9 \
    --total_length 39 \
    --img_height 96 \
    --img_width 96 \
    --lead_time_embed_dim "${LEAD_TIME_EMBED_DIM:-16}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --stride 1 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --max_samples 0 \
    --num_save_samples 24 \
    --intensity_scale "${PRECIP_SCALE}" \
    --pixel_min 0 \
    --pixel_max 255 \
    --pwv_intensity_scale "${PWV_SCALE}" \
    --pwv_pixel_min 0 \
    --pwv_pixel_max 255 \
    --pwv_invert \
    --metric_thresholds "0.5,2,5,10,30" \
    --neighborhood_metric_thresholds "16,32" \
    --neighborhood_size 5 \
    --psd_lead_minutes "60,120,180" \
    --psd_wavelengths "4,8,16,32,64" \
    --grid_km 1 \
    --frame_minutes 6 \
    --horizon_bins "0-1,1-2,2-3" \
    2>&1 | tee "${RUN_ROOT}/logs/test_pwv_v3_3h.log"
