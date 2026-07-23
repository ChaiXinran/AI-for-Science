#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the reviewed split manifest}"
: "${RADAR_CHECKPOINT:?Set RADAR_CHECKPOINT to the matched 0-2 h radar checkpoint}"
: "${PROBE_ROOT:?Set PROBE_ROOT to the diagnostic output directory}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-2026}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
REUSE_CACHE="${REUSE_CACHE:-0}"

mkdir -p "${PROBE_ROOT}/protocol"
cp code/protocols/pwv_conditional_information_probe.json "${PROBE_ROOT}/protocol/"
cp "${SPLIT_MANIFEST}" "${PROBE_ROOT}/protocol/split_manifest.json"

EXTRA_ARGS=()
if [[ "${REUSE_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse_cache)
fi

python -u code/diagnostics/pwv_conditional_probe.py \
  --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --radar_checkpoint "${RADAR_CHECKPOINT}" \
  --output_dir "${PROBE_ROOT}/seed_${SEED}" \
  --device "${DEVICE}" --require_contiguous --strict_pwv \
  --input_length 9 --total_length 29 --img_height 96 --img_width 96 \
  --ngf 32 --lead_time_embed_dim 16 --intensity_scale 35 \
  --pwv_intensity_scale 80 --pwv_invert \
  --batch_size "${BATCH_SIZE}" --probe_batch_size "${PROBE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" --epochs "${EPOCHS}" --seed "${SEED}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform \
  --thresholds 10,20 --bootstrap_repetitions 2000 \
  --minimum_csi_delta 0.003 \
  "${EXTRA_ARGS[@]}"
