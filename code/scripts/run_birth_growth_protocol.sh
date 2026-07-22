#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${RUN_ROOT:?Set RUN_ROOT to a new output directory}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEEDS="${SEEDS:-2026 2027 2028}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${RUN_ROOT}/protocol/split_manifest.json}"
SMOKE="${SMOKE:-0}"
TRAIN_LIMIT_ARGS=()
TEST_MAX_SAMPLES=0
if [[ "${SMOKE}" == "1" ]]; then
  EPOCHS=1
  SEEDS="2026"
  TRAIN_LIMIT_ARGS=(--max_train_samples 16 --max_val_samples 4)
  TEST_MAX_SAMPLES=4
fi

mkdir -p "${RUN_ROOT}/protocol"
cp code/protocols/pwv_birth_growth_v1.json "${RUN_ROOT}/protocol/"

if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  python -u code/scripts/prepare_split_manifest.py \
    --data_root "${DATA_ROOT}" \
    --pwv_root "${PWV_ROOT}" \
    --output "${SPLIT_MANIFEST}" \
    --train_ratio 0.70 --val_ratio 0.15 --seed 2026
  echo "Generated ${SPLIT_MANIFEST}. Review and merge adjacent same-storm days, then rerun."
  exit 2
fi

for SEED in ${SEEDS}; do
  SEED_ROOT="${RUN_ROOT}/seed_${SEED}"
  RADAR_CKPT="${SEED_ROOT}/checkpoints/radar/best_state_dict.ckpt"

  python -u code/train/radar.py \
    --data_root "${DATA_ROOT}" --split_manifest "${SPLIT_MANIFEST}" --require_contiguous \
    --save_dir "${SEED_ROOT}/checkpoints/radar" --readme_ckpt "${SEED_ROOT}/checkpoints/radar/model.ckpt" \
    --device "${DEVICE}" --model_name NowcastNet --input_length 9 --total_length 39 \
    --img_height 96 --img_width 96 --intensity_scale 35 --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" --seed "${SEED}" --amp \
    "${TRAIN_LIMIT_ARGS[@]}"

  python -u code/test/radar.py \
    --checkpoint "${RADAR_CKPT}" --data_root "${DATA_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --split test --max_samples "${TEST_MAX_SAMPLES}" \
    --output_dir "${SEED_ROOT}/results/radar" --device "${DEVICE}" \
    --input_length 9 --total_length 39 --img_height 96 --img_width 96 \
    --intensity_scale 35 --batch_size 1 --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --metric_thresholds 0.5,2,5,10,20,30 --neighborhood_metric_thresholds 10,20 \
    --object_thresholds 10,20 --cra_thresholds 10,20

  python -u code/train/pwv.py \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --save_dir "${SEED_ROOT}/checkpoints/birth_growth_zero" \
    --readme_ckpt "${SEED_ROOT}/checkpoints/birth_growth_zero/model.ckpt" \
    --init_radar_checkpoint "${RADAR_CKPT}" --freeze_radar_backbone \
    --device "${DEVICE}" --model_name PWVBirthGrowthNowcastNet --pwv_control zero \
    --input_length 9 --total_length 39 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope --lambda_shuffle 0 \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" "${TRAIN_LIMIT_ARGS[@]}"

  python -u code/test/pwv.py \
    --checkpoint "${SEED_ROOT}/checkpoints/birth_growth_zero/best_state_dict.ckpt" \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --split test --max_samples "${TEST_MAX_SAMPLES}" --output_dir "${SEED_ROOT}/results/birth_growth_zero" \
    --device "${DEVICE}" --model_name PWVBirthGrowthNowcastNet --pwv_control zero \
    --input_length 9 --total_length 39 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --batch_size 1 --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --metric_thresholds 0.5,2,5,10,20,30 --neighborhood_metric_thresholds 10,20 \
    --object_thresholds 10,20 --cra_thresholds 10,20

  python -u code/train/pwv.py \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --save_dir "${SEED_ROOT}/checkpoints/birth_growth" \
    --readme_ckpt "${SEED_ROOT}/checkpoints/birth_growth/model.ckpt" \
    --init_radar_checkpoint "${RADAR_CKPT}" --freeze_radar_backbone \
    --device "${DEVICE}" --model_name PWVBirthGrowthNowcastNet \
    --input_length 9 --total_length 39 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" "${TRAIN_LIMIT_ARGS[@]}"

  python -u code/test/pwv.py \
    --checkpoint "${SEED_ROOT}/checkpoints/birth_growth/best_state_dict.ckpt" \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --split test --max_samples "${TEST_MAX_SAMPLES}" --output_dir "${SEED_ROOT}/results/birth_growth" \
    --device "${DEVICE}" --model_name PWVBirthGrowthNowcastNet \
    --input_length 9 --total_length 39 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --batch_size 1 --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --metric_thresholds 0.5,2,5,10,20,30 --neighborhood_metric_thresholds 10,20 \
    --object_thresholds 10,20 --cra_thresholds 10,20
done

python -u code/report/protocol_compare.py --run_root "${RUN_ROOT}" --output "${RUN_ROOT}/protocol_summary.json"
