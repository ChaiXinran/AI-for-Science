#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the calibrated RAIN_2025_S directory}"
: "${RADAR_CHECKPOINT:?Set RADAR_CHECKPOINT to the locked radar best_state_dict.ckpt}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the locked split_manifest.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/nowcastnet_runs/radar_object_failure_attribution}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-1000}"
MODEL_NGF="${MODEL_NGF:-32}"

mkdir -p "${OUTPUT_ROOT}"

python -u code/test/radar.py \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${RADAR_CHECKPOINT}" \
  --output_dir "${OUTPUT_ROOT}/test" \
  --device "${DEVICE}" \
  --split test \
  --split_manifest "${SPLIT_MANIFEST}" \
  --require_contiguous \
  --input_length 9 \
  --total_length 39 \
  --evaluation_lead_frames 20 \
  --img_height 96 \
  --img_width 96 \
  --model_name NowcastNet \
  --ngf "${MODEL_NGF}" \
  --batch_size 1 \
  --num_workers 4 \
  --max_samples "${MAX_SAMPLES}" \
  --max_samples_strategy uniform \
  --num_save_samples 20 \
  --intensity_scale 35 \
  --metric_thresholds "1,5,10,20,30" \
  --object_thresholds "10,20,30" \
  --horizon_bins "0-1,1-2" \
  --failure_attribution \
  --attribution_thresholds "10,20,30" \
  --attribution_change_fractions "0.2,0.4,0.6" \
  --attribution_tracking_threshold_ratio 0.5 \
  --attribution_bootstrap_iterations "${BOOTSTRAP_ITERATIONS}" \
  --deterministic_noise \
  2>&1 | tee "${OUTPUT_ROOT}/test.log"

python -u code/report/radar_failure_attribution.py \
  --input "${OUTPUT_ROOT}/test/failure_attribution.json" \
  --output_dir "${OUTPUT_ROOT}/report" \
  --change_fraction 0.4 \
  --primary_horizon "1h-2h" \
  --minimum_nonadvective_miss_share 0.2 \
  2>&1 | tee "${OUTPUT_ROOT}/report.log"

echo "Failure attribution: ${OUTPUT_ROOT}/test/failure_attribution.json"
echo "Decision summary: ${OUTPUT_ROOT}/report/decision_summary.json"
