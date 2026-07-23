#!/usr/bin/env bash
set -euo pipefail

: "${PRECONDITIONING_SEED_ROOT:?Set PRECONDITIONING_SEED_ROOT to the completed seed directory}"

DEVICE="${DEVICE:-cuda:0}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
SEED="${SEED:-2026}"

python -u code/diagnostics/pwv_dynamic_residual_control.py \
  --seed_root "${PRECONDITIONING_SEED_ROOT}" \
  --device "${DEVICE}" \
  --probe_batch_size "${PROBE_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --bootstrap_repetitions "${BOOTSTRAP_REPETITIONS}" \
  --seed "${SEED}"
