#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to RAIN_2025_S}"
: "${PWV_ROOT:?Set PWV_ROOT to PWV_2025_S}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the reviewed manifest}"
: "${PILOT_ROOT:?Set PILOT_ROOT to the completed contrastive-trigger pilot directory}"

DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-512}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
SEEDS="${SEEDS:-2026 2027 2028}"

run_control() {
  local seed_id="$1"
  local result_name="$2"
  local control_name="$3"
  local seed_root="${PILOT_ROOT}/seed_${seed_id}"
  local checkpoint="${seed_root}/checkpoints/contrastive_trigger/best_state_dict.ckpt"

  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}"; exit 1; }
  python -u code/test/pwv.py \
    --checkpoint "${checkpoint}" --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" --require_contiguous --strict_pwv \
    --split "${EVAL_SPLIT}" --max_samples "${MAX_VAL_SAMPLES}" --max_samples_strategy uniform \
    --num_save_samples 0 --output_dir "${seed_root}/results/${result_name}" \
    --device "${DEVICE}" --model_name PWVContrastiveTriggerNowcastNet \
    --pwv_control "${control_name}" --input_length 9 --total_length 29 \
    --img_height 96 --img_width 96 --intensity_scale 35 \
    --pwv_intensity_scale 80 --pwv_invert \
    --pwv_tendency_windows 30,60 --pwv_tendency_mode slope \
    --pwv_candidate_threshold 0.5 --pwv_candidate_radius 4 \
    --birth_low_threshold 2 --birth_high_threshold 10 --growth_delta 5 \
    --batch_size 1 --num_workers "${NUM_WORKERS}" --seed "${seed_id}" --deterministic_noise \
    --horizon_bins 0-1,1-2,0-2 --psd_lead_minutes 60,120 --cra_lead_minutes 60,120 \
    --metric_thresholds 0.5,2,5,10,20,30 --neighborhood_metric_thresholds 10,20 \
    --object_thresholds 10,20 --cra_thresholds 10,20
}

for seed_id in ${SEEDS}; do
  echo "Running observed-input-only diagnostics for seed ${seed_id}"
  run_control "${seed_id}" pwv_temporal_reverse temporal_reverse
  run_control "${seed_id}" pwv_level_only level_only
  run_control "${seed_id}" pwv_spatial_shift spatial_shift
done

python -u code/report/protocol_compare_contrastive.py \
  --run_root "${PILOT_ROOT}" --output "${PILOT_ROOT}/protocol_summary_diagnostics.json"
