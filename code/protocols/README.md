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
