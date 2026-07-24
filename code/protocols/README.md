# Frozen PWV birth/growth protocol

This protocol removes the main comparability problems in the historical runs:

- both models target `RAIN_2025_S` at the same 35 mm/h calibration;
- a checked split manifest selects complete day directories before windows are built;
- windows with missing 6-minute frames are rejected;
- missing PWV pairs are fatal rather than silently replaced with zeros;
- each PWV model is initialized from and anchored to the matched radar-only seed;
- test sample identities are hashed and must match before results are summarized.

## Server workflow

From the repository root:

```bash
export DATA_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/RAIN_2025_S
export PWV_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/PWV_2025_S

## Radar object failure attribution

This diagnostic is the locked successor after closing dynamic-PWV modelling on
the current interpolated PNG product. It does not train a new model. It uses a
locked radar-only checkpoint to classify observed and forecast objects as
translation, rapid growth, rapid decay, birth, or split/merge, and reports
displacement, intensity, birth-existence, and full-existence oracle CSI gains.

```bash
export DATA_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/RAIN_2025_S
export RADAR_CHECKPOINT=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1_radar_gate/checkpoints/radar/best_state_dict.ckpt
export SPLIT_MANIFEST=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1/protocol/split_manifest.json
export OUTPUT_ROOT=/root/autodl-tmp/nowcastnet_runs/radar_object_failure_attribution
export MODEL_NGF=32
export MAX_SAMPLES=0
bash code/scripts/run_radar_failure_attribution.sh
```

`MAX_SAMPLES=0` evaluates all locked test windows. The primary artifact is
`test/failure_attribution.json`; the report folder contains four PNG figures
and `decision_summary.json`. Oracle gains are diagnostic upper bounds, not
achievable forecast scores.
export RUN_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1
export DEVICE=cuda:0
export BATCH_SIZE=8
export EPOCHS=60
export NUM_WORKERS=8
export SEEDS="2026 2027 2028"

bash code/scripts/run_birth_growth_protocol.sh
```

The first invocation creates `RUN_ROOT/protocol/split_manifest.json` and exits.
Inspect the train/validation and validation/test boundaries. If adjacent dates
belong to the same storm, move them to the same split. Commit or archive this
exact manifest and then rerun the command.

For the May--August 2025 North China dataset, the reviewed boundaries are
2025-07-22 and 2025-08-14. Regenerate and lock the manifest without hand-editing:

```bash
python -u code/scripts/prepare_split_manifest.py \
  --data_root "${DATA_ROOT}" --pwv_root "${PWV_ROOT}" \
  --output "${RUN_ROOT}/protocol/split_manifest.json" \
  --train_end 20250722 --val_end 20250814 --seed 2026
```

This yields 83 train days, 23 validation days, and 17 test days. The July
23--30 extreme-rain event and the August 12--14 process remain wholly in the
validation block; the test block begins on August 15.

Before the full run, use a separate output folder for an end-to-end smoke test.
Smoke mode uses 64 uniformly spaced training windows and 32 validation/test
windows, so rare-event metrics are less likely to be empty:

```bash
export RUN_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1_smoke
export SPLIT_MANIFEST=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1/protocol/split_manifest.json
export SMOKE=1
bash code/scripts/run_birth_growth_protocol.sh
unset SMOKE
```

## Quick server smoke test

Use one seed and small sample limits by invoking the training scripts directly;
do not treat smoke-test metrics as scientific evidence. The full protocol script
intentionally has no sample-limit shortcut.

## Outputs

Each seed contains matched radar-only, zero-PWV Birth/Growth control, and real-PWV
Birth/Growth checkpoint/result directories. Every checkpoint folder
contains `data_manifest.json`; every result folder contains an independent test
manifest. The final `protocol_summary.json` is generated only when sample hashes
match exactly.

## Signed PWV calibrator successor

The positive-only `contrastive_trigger` branch is archived as a no-go. Its
successor is goal-named rather than version-numbered:

- protocol: `code/protocols/pwv_signed_calibrator_pilot.json`
- runner: `code/scripts/run_signed_calibrator_pilot.sh`
- model: `PWVSignedCalibratorNowcastNet`

The runner first retrains a matched radar baseline with the same 9-frame input,
20-frame (0--2 h) output, split, and 2048/512 budget. It then runs the
train-only PWV climatology/support audit once and trains static,
spatial-control, and temporal-tendency heads. It evaluates null, level-only,
reverse, and shift controls from the static checkpoint and produces paired
day-cluster bootstrap deltas. A 0--3 h radar checkpoint is not shape-compatible
and must not be reused.

Server example:

```bash
export DATA_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/RAIN_2025_S
export PWV_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/PWV_2025_S
export SPLIT_MANIFEST=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1/protocol/split_manifest.json
export PILOT_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_signed_calibrator_pilot
export DEVICE=cuda:0 BATCH_SIZE=8 NUM_WORKERS=8 EPOCHS=10 SEED=2026
bash code/scripts/run_signed_calibrator_pilot.sh 2>&1 | tee "${PILOT_ROOT}/run.log"
```

The one-seed run can only promote the design to three-seed replication. It
must not be presented as final test evidence.

## PWV latent-state fusion successor

The signed recursive-source pilot failed its CSI/FAR/MAE gate. The successor
therefore treats PWV as an encoded atmospheric state and fuses it once at the
generative bottleneck. PWV never enters the recursive source or motion
equations, and `Zero-PWV` is not used as a separate experiment.

- protocol: `code/protocols/pwv_latent_state_fusion_pilot.json`
- runner: `code/scripts/run_latent_state_fusion_pilot.sh`
- model: `PWVLatentFusionNowcastNet`
- variants: continued radar-only, aligned PWV, and independently/randomly
  displaced PWV during training (deterministic half-domain shift at evaluation)

The three variants start from the same matched 0--2 h radar checkpoint and
receive the same additional epoch budget. The radar path is fine-tuned at
one-tenth the learning rate used by the new PWV encoder and fusion modules.

```bash
export DATA_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/RAIN_2025_S
export PWV_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/PWV_2025_S
export SPLIT_MANIFEST=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1/protocol/split_manifest.json
export RADAR_INIT_CKPT=/root/autodl-tmp/nowcastnet_runs/pwv_signed_calibrator_pilot/seed_2026/checkpoints/radar/best_state_dict.ckpt
export PILOT_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_latent_state_fusion_pilot
export DEVICE=cuda:0 BATCH_SIZE=8 NUM_WORKERS=8 EPOCHS=10 SEED=2026
bash code/scripts/run_latent_state_fusion_pilot.sh 2>&1 | tee "${PILOT_ROOT}/run.log"
```

## PWV conditional-information probe

The latent-state pilot is closed because its same-checkpoint forecast was
insensitive to aligned versus shifted PWV. Before another end-to-end model is
allowed, this diagnostic tests whether PWV contains conditional information
after a trained radar latent is frozen.

- protocol: `code/protocols/pwv_conditional_information_probe.json`
- runner: `code/scripts/run_conditional_information_probe.sh`
- output: `seed_2026/conditional_probe_summary.json`

The radar-only and PWV probes have identical parameter counts. The aligned PWV
checkpoint is also evaluated with spatially shifted and cross-event PWV.
Probability thresholds are selected on held-out training days and fixed before
validation.

```bash
export DATA_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/RAIN_2025_S
export PWV_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/PWV_2025_S
export SPLIT_MANIFEST=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1/protocol/split_manifest.json
export RADAR_CHECKPOINT=/root/autodl-tmp/nowcastnet_runs/pwv_latent_state_fusion_pilot/seed_2026/checkpoints/radar_only/best_state_dict.ckpt
export PROBE_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_conditional_information_probe
export DEVICE=cuda:0 BATCH_SIZE=8 PROBE_BATCH_SIZE=32 NUM_WORKERS=8 EPOCHS=20 SEED=2026
bash code/scripts/run_conditional_information_probe.sh 2>&1 | tee "${PROBE_ROOT}/run.log"
```

## Causal PWV preconditioning probe

The six-minute PWV rasters are predominantly interpolated from an approximately
30-minute product. This successor therefore keeps the existing nine-frame
radar input but reads seven causal PWV anchors over three hours. The latest PWV
anchor is always at or before the final observed radar timestamp.

- protocol: `code/protocols/pwv_causal_preconditioning_probe.json`
- runner: `code/scripts/run_pwv_preconditioning_probe.sh`
- output: `seed_2026/preconditioning_probe_summary.json`

It trains parameter-matched radar-only, short-interpolated-PWV, and
causal-long-PWV probes. The long-PWV checkpoint is additionally tested with
spatial shift, cross-event replacement, and tendency reversal. Metrics are
reported for all tiles, a radar-observable weak-echo/nondecreasing stratum, and
a diagnostic radar-quiet stratum.

```bash
export DATA_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/RAIN_2025_S
export PWV_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S/DATA_2025_S/PWV_2025_S
export SPLIT_MANIFEST=/root/autodl-tmp/nowcastnet_runs/pwv_birth_growth_v1/protocol/split_manifest.json
export RADAR_CHECKPOINT=/root/autodl-tmp/nowcastnet_runs/pwv_latent_state_fusion_pilot/seed_2026/checkpoints/radar_only/best_state_dict.ckpt
export PROBE_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_causal_preconditioning_probe
export DEVICE=cuda:0 BATCH_SIZE=8 PROBE_BATCH_SIZE=32 NUM_WORKERS=8 EPOCHS=20 SEED=2026
mkdir -p "${PROBE_ROOT}"
bash code/scripts/run_pwv_preconditioning_probe.sh 2>&1 | tee "${PROBE_ROOT}/run.log"
```

### Same-checkpoint PWV attribution

After the causal preconditioning probe finishes, this inference-only diagnostic
decomposes PWV into static fit-day climatology, an event-level scalar moisture
offset, and an event-specific spatial anomaly. It reuses the completed cache,
the trained long-PWV probe, and its frozen calibration thresholds. No model is
retrained.

```bash
export PRECONDITIONING_SEED_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_causal_preconditioning_probe/seed_2026
mkdir -p "${PRECONDITIONING_SEED_ROOT}"
bash code/scripts/run_pwv_preconditioning_attribution.sh \
  2>&1 | tee "${PRECONDITIONING_SEED_ROOT}/attribution.log"
```

The result is
`seed_2026/preconditioning_attribution_summary.json`.

### Fair static-climatology versus dynamic-PWV control

This is the final trained attribution gate. Both probes receive the identical
fit-day static PWV climatology channel. Only the dynamic probe receives the
event scalar moisture state and zero-mean spatial residuals. Both probes use
the same architecture, initialization seed, fit/calibration days, and
optimization budget.

```bash
export PRECONDITIONING_SEED_ROOT=/root/autodl-tmp/nowcastnet_runs/pwv_causal_preconditioning_probe/seed_2026
mkdir -p "${PRECONDITIONING_SEED_ROOT}/fair_dynamic_residual_control"
bash code/scripts/run_pwv_fair_dynamic_control.sh \
  2>&1 | tee "${PRECONDITIONING_SEED_ROOT}/fair_dynamic_residual_control/run.log"
```

The result is
`seed_2026/fair_dynamic_residual_control/fair_dynamic_residual_summary.json`.
