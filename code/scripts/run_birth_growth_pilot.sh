#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the reviewed manifest}"
: "${RADAR_GATE_ROOT:?Set RADAR_GATE_ROOT to the completed radar gate run}"
: "${PILOT_ROOT:?Set PILOT_ROOT to a new pilot output directory}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-2026}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-512}"

SEED_ROOT="${PILOT_ROOT}/seed_${SEED}"
RADAR_CKPT="${RADAR_GATE_ROOT}/checkpoints/radar/best_state_dict.ckpt"
RADAR_RESULTS="${RADAR_GATE_ROOT}/results/radar"

[[ -f "${RADAR_CKPT}" ]] || { echo "Missing ${RADAR_CKPT}"; exit 1; }
[[ -f "${RADAR_RESULTS}/metrics.json" ]] || { echo "Missing radar metrics"; exit 1; }
[[ -f "${RADAR_RESULTS}/data_manifest.json" ]] || { echo "Missing radar data manifest"; exit 1; }

mkdir -p "${SEED_ROOT}/results/radar"
cp "${RADAR_RESULTS}/metrics.json" "${SEED_ROOT}/results/radar/metrics.json"
cp "${RADAR_RESULTS}/data_manifest.json" "${SEED_ROOT}/results/radar/data_manifest.json"

run_head() {
  local name="$1"
  local control="$2"

  python -u code/train/pwv.py \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --save_dir "${SEED_ROOT}/checkpoints/${name}" \
    --readme_ckpt "${SEED_ROOT}/checkpoints/${name}/model.ckpt" \
    --init_radar_checkpoint "${RADAR_CKPT}" --freeze_radar_backbone \
    --device "${DEVICE}" --model_name PWVBirthGrowthNowcastNet --pwv_control "${control}" \
    --input_length 9 --total_length 39 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope --lambda_shuffle 0 \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --birth_loss_normalization class_balanced --source_inactive_weight 0.1 \
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform

  python -u code/test/pwv.py \
    --checkpoint "${SEED_ROOT}/checkpoints/${name}/best_state_dict.ckpt" \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --split test --max_samples "${MAX_TEST_SAMPLES}" --max_samples_strategy uniform \
    --num_save_samples 10 --output_dir "${SEED_ROOT}/results/${name}" \
    --device "${DEVICE}" --model_name PWVBirthGrowthNowcastNet --pwv_control "${control}" \
    --input_length 9 --total_length 39 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --batch_size 1 --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --metric_thresholds 0.5,2,5,10,20,30 --neighborhood_metric_thresholds 10,20 \
    --object_thresholds 10,20 --cra_thresholds 10,20
}

run_head birth_growth_zero zero
run_head birth_growth real

python -u code/report/protocol_compare.py \
  --run_root "${PILOT_ROOT}" --output "${PILOT_ROOT}/protocol_summary.json"
