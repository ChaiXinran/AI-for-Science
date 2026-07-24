#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${RADAR_CHECKPOINT:?Set RADAR_CHECKPOINT to the locked 9-input/30-output radar checkpoint}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the locked split_manifest.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/nowcastnet_runs/pwv_survival_intensity_adapter_pilot}"
DEVICE="${DEVICE:-cuda:0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
EPOCHS="${EPOCHS:-12}"
ADAPTER_BATCH_SIZE="${ADAPTER_BATCH_SIZE:-4}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-1000}"

export OMP_NUM_THREADS="${PWV_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${PWV_MKL_NUM_THREADS:-1}"
mkdir -p "${OUTPUT_ROOT}"

python -u code/experiments/pwv_survival_intensity_adapter.py \
  --data_root "${DATA_ROOT}" \
  --pwv_root "${PWV_ROOT}" \
  --radar_checkpoint "${RADAR_CHECKPOINT}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --output_dir "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --seed 2026 \
  --input_length 9 \
  --total_length 39 \
  --evaluation_lead_frames 20 \
  --img_height 96 \
  --img_width 96 \
  --model_name NowcastNet \
  --ngf 32 \
  --lead_time_embed_dim 16 \
  --batch_size 1 \
  --adapter_batch_size "${ADAPTER_BATCH_SIZE}" \
  --num_workers 4 \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" \
  --max_samples_strategy uniform \
  --intensity_scale 35 \
  --pwv_intensity_scale 80 \
  --pwv_history_minutes 180 \
  --pwv_anchor_minutes 30 \
  --require_contiguous \
  --strict_pwv \
  --hidden_channels 32 \
  --max_correction 12 \
  --candidate_threshold 0.5 \
  --candidate_radius 3 \
  --epochs "${EPOCHS}" \
  --learning_rate 3e-4 \
  --lambda_soft_csi 0.5 \
  --lambda_far 0.1 \
  --lambda_correction 0.002 \
  --thresholds 10,20,30 \
  --selection_thresholds 10,20 \
  --bootstrap_repetitions "${BOOTSTRAP_REPETITIONS}" \
  --minimum_csi_delta 0.003 \
  2>&1 | tee "${OUTPUT_ROOT}/run.log"

echo "Metrics: ${OUTPUT_ROOT}/metrics.json"
echo "History: ${OUTPUT_ROOT}/train_history.json"
