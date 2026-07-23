#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the reviewed manifest}"
: "${PILOT_ROOT:?Set PILOT_ROOT to a new signed-calibrator pilot directory}"

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
PROTOCOL_ROOT="${PILOT_ROOT}/protocol"
STAGE0_ROOT="${PROTOCOL_ROOT}/stage0"
CLIMATOLOGY_PATH="${STAGE0_ROOT}/pwv_train_climatology.npz"
RADAR_DIR="${SEED_ROOT}/checkpoints/radar"
RADAR_CKPT="${RADAR_DIR}/best_state_dict.ckpt"
STATIC_DIR="${SEED_ROOT}/checkpoints/static_real"
SPATIAL_DIR="${SEED_ROOT}/checkpoints/spatial_control"
TENDENCY_DIR="${SEED_ROOT}/checkpoints/tendency_real"

mkdir -p "${PROTOCOL_ROOT}" "${RADAR_DIR}" "${STATIC_DIR}" "${SPATIAL_DIR}" "${TENDENCY_DIR}"
cp code/protocols/pwv_signed_calibrator_pilot.json "${PROTOCOL_ROOT}/"
cp "${SPLIT_MANIFEST}" "${PROTOCOL_ROOT}/split_manifest.json"

if [[ ! -f "${CLIMATOLOGY_PATH}" ]]; then
  python -u code/scripts/audit_pwv_stage0.py \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --output_dir "${STAGE0_ROOT}" \
    --input_length 9 --total_length 29 --frame_minutes 6 \
    --thresholds 10,20,30 --horizon_bins 0-1,1-2,0-2 \
    --rain_intensity_scale 35 --pwv_intensity_scale 80 \
    --climatology_frame_stride 6 --transition_samples_per_event 12 \
    --feature_windows_per_event 6
fi

[[ -f "${CLIMATOLOGY_PATH}" ]] || { echo "Missing ${CLIMATOLOGY_PATH}"; exit 1; }

if [[ "${RETEST_ONLY}" != "1" ]]; then
  python -u code/train/radar.py \
    --data_root "${DATA_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --require_contiguous --save_dir "${RADAR_DIR}" \
    --readme_ckpt "${RADAR_DIR}/model.ckpt" --device "${DEVICE}" \
    --input_length 9 --total_length 29 --img_height 96 --img_width 96 \
    --ngf 32 --intensity_scale 35 --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform
fi

[[ -f "${RADAR_CKPT}" ]] || {
  echo "Missing matched 0-2 h radar checkpoint: ${RADAR_CKPT}"
  echo "Do not reuse the old total_length=39 (0-3 h) radar checkpoint."
  exit 1
}

train_head() {
  local save_dir="$1"
  local control="$2"
  local tendency_flag="${3:-0}"
  local extra_args=()
  if [[ "${tendency_flag}" == "1" ]]; then
    extra_args+=(--signed_use_tendency)
  fi
  python -u code/train/pwv_signed.py \
    --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --save_dir "${save_dir}" --readme_ckpt "${save_dir}/model.ckpt" \
    --init_radar_checkpoint "${RADAR_CKPT}" --freeze_radar_backbone \
    --pwv_climatology_path "${CLIMATOLOGY_PATH}" \
    --model_name PWVSignedCalibratorNowcastNet --pwv_control "${control}" \
    --device "${DEVICE}" --input_length 9 --total_length 29 \
    --img_height 96 --img_width 96 --ngf 32 --evo_base_channels 32 \
    --pwv_base_channels 24 --fusion_channels 32 --lead_time_embed_dim 16 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_candidate_threshold 0.5 --pwv_candidate_radius 4 \
    --signed_residual_scale 0.25 --calibration_thresholds 10,20 \
    --calibration_temperature 1.0 --lambda_calibration 1.0 \
    --lambda_forecast 0.1 --lambda_false_alarm 0.1 \
    --lambda_signed_contribution 0.02 --early_stop_patience 3 \
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform \
    "${extra_args[@]}"
}

if [[ "${RETEST_ONLY}" != "1" ]]; then
  train_head "${STATIC_DIR}" real 0
  train_head "${SPATIAL_DIR}" spatial_shift 0
  train_head "${TENDENCY_DIR}" real 1
fi

STATIC_CKPT="${STATIC_DIR}/best_state_dict.ckpt"
SPATIAL_CKPT="${SPATIAL_DIR}/best_state_dict.ckpt"
TENDENCY_CKPT="${TENDENCY_DIR}/best_state_dict.ckpt"
for checkpoint in "${STATIC_CKPT}" "${SPATIAL_CKPT}" "${TENDENCY_CKPT}"; do
  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}"; exit 1; }
done

evaluate_head() {
  local checkpoint="$1"
  local result_name="$2"
  local control="$3"
  local tendency_flag="${4:-0}"
  local extra_args=()
  if [[ "${tendency_flag}" == "1" ]]; then
    extra_args+=(--signed_use_tendency)
  fi
  python -u code/test/pwv.py \
    --checkpoint "${checkpoint}" --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --split "${EVAL_SPLIT}" --max_samples "${MAX_VAL_SAMPLES}" \
    --max_samples_strategy uniform --num_save_samples 6 \
    --output_dir "${SEED_ROOT}/results/${result_name}" \
    --device "${DEVICE}" --model_name PWVSignedCalibratorNowcastNet \
    --pwv_control "${control}" --pwv_climatology_path "${CLIMATOLOGY_PATH}" \
    --input_length 9 --total_length 29 --img_height 96 --img_width 96 \
    --ngf 32 --evo_base_channels 32 --pwv_base_channels 24 \
    --fusion_channels 32 --lead_time_embed_dim 16 \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --pwv_candidate_threshold 0.5 --pwv_candidate_radius 4 \
    --signed_residual_scale 0.25 --batch_size 1 \
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" --deterministic_noise \
    --horizon_bins 0-1,1-2,0-2 --metric_thresholds 0.5,2,5,10,20,30 \
    --neighborhood_metric_thresholds 10,20 --object_thresholds 10,20 \
    --cra_thresholds 10,20 --psd_lead_minutes 60,120 \
    --cra_lead_minutes 60,120 \
    "${extra_args[@]}"
}

evaluate_head "${STATIC_CKPT}" static_real real 0
evaluate_head "${STATIC_CKPT}" static_null zero 0
evaluate_head "${STATIC_CKPT}" static_level level_only 0
evaluate_head "${STATIC_CKPT}" static_reverse temporal_reverse 0
evaluate_head "${STATIC_CKPT}" static_shift spatial_shift 0
evaluate_head "${SPATIAL_CKPT}" spatial_control spatial_shift 0
evaluate_head "${TENDENCY_CKPT}" tendency_real real 1

python -u code/report/protocol_compare_signed.py \
  --seed_root "${SEED_ROOT}" \
  --output "${SEED_ROOT}/signed_protocol_summary.json" \
  --thresholds 10,20 --bootstrap_repetitions 2000 --bootstrap_seed "${SEED}" \
  --minimum_csi_delta 0.003 --maximum_far_delta 0.005 \
  --maximum_relative_mae_increase 0.005
