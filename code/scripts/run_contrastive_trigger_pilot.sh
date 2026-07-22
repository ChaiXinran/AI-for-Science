#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the reviewed manifest}"
: "${PILOT_ROOT:?Set PILOT_ROOT to a new contrastive-trigger pilot directory}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-2026}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
RETEST_ONLY="${RETEST_ONLY:-0}"

SEED_ROOT="${PILOT_ROOT}/seed_${SEED}"
RADAR_DIR="${SEED_ROOT}/checkpoints/radar"
TRIGGER_DIR="${SEED_ROOT}/checkpoints/contrastive_trigger"
RADAR_CKPT="${RADAR_DIR}/best_state_dict.ckpt"
TRIGGER_CKPT="${TRIGGER_DIR}/best_state_dict.ckpt"

mkdir -p "${PILOT_ROOT}/protocol" "${RADAR_DIR}" "${TRIGGER_DIR}"
cp code/protocols/pwv_contrastive_trigger_pilot.json "${PILOT_ROOT}/protocol/"

if [[ "${RETEST_ONLY}" != "1" ]]; then
  python -u code/train/radar.py \
    --data_root "${DATA_ROOT}" --split_manifest "${SPLIT_MANIFEST}" --require_contiguous \
    --save_dir "${RADAR_DIR}" --readme_ckpt "${RADAR_DIR}/model.ckpt" \
    --device "${DEVICE}" --model_name NowcastNet \
    --input_length 9 --total_length 29 --img_height 96 --img_width 96 \
    --intensity_scale 35 --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" --max_val_samples "${MAX_VAL_SAMPLES}" \
    --max_samples_strategy uniform

  python -u code/train/pwv.py \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --save_dir "${TRIGGER_DIR}" --readme_ckpt "${TRIGGER_DIR}/model.ckpt" \
    --init_radar_checkpoint "${RADAR_CKPT}" --freeze_radar_backbone \
    --device "${DEVICE}" --model_name PWVContrastiveTriggerNowcastNet --pwv_control real \
    --input_length 9 --total_length 29 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope \
    --pwv_candidate_threshold 0.5 --pwv_candidate_radius 4 \
    --lambda_shuffle 0.05 --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --birth_loss_normalization class_balanced --source_inactive_weight 0.1 \
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform
fi

[[ -f "${RADAR_CKPT}" ]] || { echo "Missing ${RADAR_CKPT}"; exit 1; }
[[ -f "${TRIGGER_CKPT}" ]] || { echo "Missing ${TRIGGER_CKPT}"; exit 1; }

python -u code/test/radar.py \
  --checkpoint "${RADAR_CKPT}" --data_root "${DATA_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --split "${EVAL_SPLIT}" \
  --max_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform \
  --num_save_samples 10 --output_dir "${SEED_ROOT}/results/radar" \
  --device "${DEVICE}" --input_length 9 --total_length 29 \
  --img_height 96 --img_width 96 --intensity_scale 35 \
  --batch_size 1 --num_workers "${NUM_WORKERS}" --seed "${SEED}" --deterministic_noise \
  --horizon_bins 0-1,1-2,0-2 --psd_lead_minutes 60,120 --cra_lead_minutes 60,120 \
  --metric_thresholds 0.5,2,5,10,20,30 --neighborhood_metric_thresholds 10,20 \
  --object_thresholds 10,20 --cra_thresholds 10,20

test_trigger() {
  local name="$1"
  local control="$2"
  python -u code/test/pwv.py \
    --checkpoint "${TRIGGER_CKPT}" --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --split "${EVAL_SPLIT}" --max_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform \
    --num_save_samples 10 --output_dir "${SEED_ROOT}/results/${name}" \
    --device "${DEVICE}" --model_name PWVContrastiveTriggerNowcastNet --pwv_control "${control}" \
    --input_length 9 --total_length 29 --img_height 96 --img_width 96 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope \
    --pwv_candidate_threshold 0.5 --pwv_candidate_radius 4 \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --batch_size 1 --num_workers "${NUM_WORKERS}" --seed "${SEED}" --deterministic_noise \
    --horizon_bins 0-1,1-2,0-2 --psd_lead_minutes 60,120 --cra_lead_minutes 60,120 \
    --metric_thresholds 0.5,2,5,10,20,30 --neighborhood_metric_thresholds 10,20 \
    --object_thresholds 10,20 --cra_thresholds 10,20
}

test_trigger pwv_real real
test_trigger pwv_null zero
test_trigger pwv_temporal_reverse temporal_reverse

python -u code/report/protocol_compare_contrastive.py \
  --run_root "${PILOT_ROOT}" --output "${PILOT_ROOT}/protocol_summary.json"
