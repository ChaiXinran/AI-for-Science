#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the reviewed manifest}"
: "${RADAR_INIT_CKPT:?Set RADAR_INIT_CKPT to the matched 0-2 h radar checkpoint}"
: "${PILOT_ROOT:?Set PILOT_ROOT to the latent-state fusion pilot directory}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-2026}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
RETEST_ONLY="${RETEST_ONLY:-0}"

SEED_ROOT="${PILOT_ROOT}/seed_${SEED}"
RADAR_DIR="${SEED_ROOT}/checkpoints/radar_only"
ALIGNED_DIR="${SEED_ROOT}/checkpoints/pwv_aligned"
DISPLACED_DIR="${SEED_ROOT}/checkpoints/pwv_displaced"
mkdir -p "${PILOT_ROOT}/protocol" "${RADAR_DIR}" "${ALIGNED_DIR}" "${DISPLACED_DIR}"
cp code/protocols/pwv_latent_state_fusion_pilot.json "${PILOT_ROOT}/protocol/"
cp "${SPLIT_MANIFEST}" "${PILOT_ROOT}/protocol/split_manifest.json"

[[ -f "${RADAR_INIT_CKPT}" ]] || {
  echo "Missing matched 0-2 h radar checkpoint: ${RADAR_INIT_CKPT}"
  exit 1
}

if [[ "${RETEST_ONLY}" != "1" ]]; then
  python -u code/train/radar.py \
    --data_root "${DATA_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --require_contiguous --save_dir "${RADAR_DIR}" \
    --readme_ckpt "${RADAR_DIR}/model.ckpt" \
    --init_generator "${RADAR_INIT_CKPT}" --device "${DEVICE}" \
    --input_length 9 --total_length 29 --img_height 96 --img_width 96 \
    --ngf 32 --intensity_scale 35 --lr_g 1e-5 \
    --matched_discriminator_seed "$((SEED + 1000))" \
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform

  train_pwv() {
    local save_dir="$1"
    local control="$2"
    python -u code/train/pwv_latent.py \
      --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
      --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
      --save_dir "${save_dir}" --readme_ckpt "${save_dir}/model.ckpt" \
      --init_radar_checkpoint "${RADAR_INIT_CKPT}" \
      --model_name PWVLatentFusionNowcastNet --pwv_control "${control}" \
      --device "${DEVICE}" --input_length 9 --total_length 29 \
      --img_height 96 --img_width 96 --ngf 32 --evo_base_channels 32 \
      --lead_time_embed_dim 16 --pwv_latent_channels 8 \
      --pwv_latent_heads 4 --intensity_scale 35 \
      --pwv_intensity_scale 80 --pwv_invert \
      --lambda_pwv_aux 0.1 --lr_g 1e-4 --radar_lr_scale 0.1 \
      --early_stop_patience "${EPOCHS}" \
      --matched_discriminator_seed "$((SEED + 1000))" \
      --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" \
      --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
      --max_train_samples "${MAX_TRAIN_SAMPLES}" \
      --max_val_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform
  }
  train_pwv "${ALIGNED_DIR}" real
  train_pwv "${DISPLACED_DIR}" spatial_shift
fi

RADAR_CKPT="${RADAR_DIR}/best_state_dict.ckpt"
ALIGNED_CKPT="${ALIGNED_DIR}/best_state_dict.ckpt"
DISPLACED_CKPT="${DISPLACED_DIR}/best_state_dict.ckpt"
for checkpoint in "${RADAR_CKPT}" "${ALIGNED_CKPT}" "${DISPLACED_CKPT}"; do
  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}"; exit 1; }
done

COMMON_METRICS=(
  --split val --max_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform
  --num_save_samples 6 --device "${DEVICE}" --input_length 9 --total_length 29
  --img_height 96 --img_width 96 --ngf 32 --lead_time_embed_dim 16
  --intensity_scale 35 --batch_size 1 --num_workers "${NUM_WORKERS}"
  --seed "${SEED}" --deterministic_noise --horizon_bins 0-1,1-2,0-2
  --metric_thresholds 0.5,2,5,10,20,30
  --neighborhood_metric_thresholds 10,20 --object_thresholds 10,20
  --cra_thresholds 10,20 --psd_lead_minutes 60,120
  --cra_lead_minutes 60,120
)

python -u code/test/radar.py \
  --checkpoint "${RADAR_CKPT}" --data_root "${DATA_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" --require_contiguous \
  --output_dir "${SEED_ROOT}/results/radar_only" \
  "${COMMON_METRICS[@]}"

evaluate_pwv() {
  local checkpoint="$1"
  local result_name="$2"
  local control="$3"
  python -u code/test/pwv.py \
    --checkpoint "${checkpoint}" --data_root "${DATA_ROOT}" \
    --pwv_root "${PWV_ROOT}" --split_manifest "${SPLIT_MANIFEST}" \
    --require_contiguous --strict_pwv \
    --output_dir "${SEED_ROOT}/results/${result_name}" \
    --model_name PWVLatentFusionNowcastNet --pwv_control "${control}" \
    --evo_base_channels 32 --pwv_latent_channels 8 \
    --pwv_latent_heads 4 --pwv_intensity_scale 80 --pwv_invert \
    "${COMMON_METRICS[@]}"
}

evaluate_pwv "${ALIGNED_CKPT}" pwv_aligned real
evaluate_pwv "${DISPLACED_CKPT}" pwv_displaced spatial_shift

python -u code/report/protocol_compare_latent.py \
  --seed_root "${SEED_ROOT}" \
  --output "${SEED_ROOT}/latent_protocol_summary.json" \
  --thresholds 10,20 --bootstrap_repetitions 2000 \
  --bootstrap_seed "${SEED}" --minimum_csi_delta 0.003 \
  --maximum_far_delta 0.005 --maximum_relative_mae_increase 0.005
