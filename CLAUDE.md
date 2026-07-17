# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **NowcastNet** implementation for precipitation nowcasting — predicting future radar rainfall from a sequence of past radar frames using deep learning. The codebase extends the original NowcastNet inference-only capsule with custom training pipelines, PWV (Precipitable Water Vapor) coupled model variants, and comprehensive evaluation tooling.

## Environment & Execution

All commands run from `code/`. Use the `nowcast` conda environment:

```bash
conda activate nowcast
cd code/
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Quick smoke test (GPU, WSL)
```bash
python -u train_adversarial_custom.py \
  --data_root ../data/DATA_2025_S/RADAR_2025_S \
  --save_dir ../checkpoints/smoke_adv_gpu \
  --readme_ckpt ../checkpoints/smoke_mrms_model.ckpt \
  --device cuda:0 --img_height 96 --img_width 96 \
  --batch_size 1 --epochs 1 --num_workers 0 \
  --max_train_samples 8 --max_val_samples 2 --disc_channels 8 --amp
```

### Full experiments use shell scripts

All experiment scripts live in `code/`. The naming convention is:

| Prefix | Purpose |
|--------|---------|
| `quick_train_*.sh` / `quick_test_*.sh` | Fast exploration runs (few samples) |
| `server_train_*.sh` / `server_test_*.sh` | Full-sample runs on AutoDL 5090 |
| `server_run_*.sh` | Combined train+test+report for a variant |

Key scripts:
- **Radar-only baseline**: `server_train_radar_3h.sh` / `server_test_radar_3h.sh`
- **PWV V2**: `server_train_pwv_v2_3h.sh` / `server_test_pwv_v2_3h.sh`
- **PWV V3** (false-alarm control): `server_train_pwv_v3_3h.sh` / `server_test_pwv_v3_3h.sh`
- **PWV V4** (convolutional LSTM): `server_train_pwv_v4_3h.sh` / `server_test_pwv_v4_3h.sh`
- **V3 Object** (object-aware losses): `server_train_pwv_v3_object_3h.sh`
- **V3 Object Consistency** (temporal object consistency): `server_train_pwv_v3_object_consistency_3h.sh`
- **V3 Object Consistency + Tendency + FACL**: `server_run_pwv_v3_object_consistency_tendency_facl_3h.sh`
- **Orchestrate all**: `server_run_all_3h.sh` — runs all three main variants end-to-end

### Report generation
```bash
python make_server_3h_report.py --run_root /root/autodl-tmp/nowcastnet_runs/north_china_3h_physical
python make_horizon_report.py     # for horizon-comparison plots
python make_quick_report.py
```

### Model names for `--model_name`

Registered in `nowcasting/models/registry.py`:
- `NowcastNet` / `nowcasting` — original radar-only model
- `PWVCoupledNowcastNet` — V1: PWV source decomposed via coupling field, zero-gated evolution nets
- `PWVCoupledNowcastNetV2` — V2: PWV features (value, anomaly, delta, gradient + tendency), lightweight U-Net coupling
- `PWVCoupledNowcastNetV3` — V3: adds feature-space gating and PWV support gate with false-alarm losses
- `PWVCoupledNowcastNetV3Object` — V3 + object-aware segmentation head and losses
- `PWVCoupledNowcastNetV4` — V4: replaces evolution with convolutional LSTM

## High-Level Architecture

The model follows a **two-stage evolution-generative** design from the NowcastNet paper:

### Stage 1: Evolution Network (`layers/evolution/`)
- Takes the input radar sequence and predicts per-pixel **motion fields** (optical flow) and **intensity residuals**
- Warps the last observed frame forward using the predicted motion, then adds intensity increments
- Produces a deterministic "first guess" future sequence

### Stage 2: Generative Network (`layers/generation/`)
- Takes the evolution output + input frames through a convolutional encoder
- Injects learned noise via a `Noise_Projector`
- The decoder refines the evolution output, adding stochastic detail and correcting errors

### PWV-Coupled Variants (V1–V4)

All share the same decomposition pattern on top of the base model:
```
s_t = s_t^radar + C_t(x, y) * s_t^PWV
```
Where `C_t` is a learned spatiotemporal coupling field (visualized as `c_*.png` in test output).

Key differences:
- **V1**: Three separate `Evolution_Network` instances (radar, PWV source, coupling gate), all zero-gated
- **V2**: Replaces coupling network with `LightweightUNet`; PWV input expanded to multi-feature maps via `pwv_features.py` (raw value, anomaly, temporal change, spatial gradient, configurable tendency windows)
- **V3**: Adds **feature-space gating** in the fusion (`F_fuse = Z_r + C_f * A_pwv`) and an explicit **PWV support gate** `S_pwv` trained to stay low in dry regions. Additional losses: `false_alarm_loss`, `support_dry_loss`, `support_l1`
- **V4**: Replaces `Evolution_Network` with a convolutional LSTM backbone. Source decomposition and coupling remain similar to V2
- **V3Object**: Adds an `ObjectHead` (`nowcasting/models/object_head.py`) for segmentation-style object-aware prediction, with object consistency losses

### Training Pipeline

Training uses **adversarial training** with:
- **Generator**: The NowcastNet model (any variant)
- **Temporal Discriminator** (`nowcasting/models/temporal_discriminator.py`): Multi-branch 3D CNN that scores temporal coherence at multiple timescales
- **Losses**: Weighted L1 (rain-rate-weighted), evolution loss, advection loss, motion regularization, multi-scale pooling loss, adversarial loss, plus variant-specific losses and optional FACL (Frequency-Aware Contrastive Learning, `nowcasting/facl.py`)

Mixed precision (AMP) is supported via `--amp`. PWV V2/V3/V4 run in full precision by default for numerical stability.

### Data Pipeline

- **`PngSequenceDataset`** (`data_provider/custom_png.py`): Sliding-window loader over chronological PNG folders organized as `YYYYMM/YYYYMMDD/*.png`. Each sample is `input_length` + `pred_length` consecutive frames
- Images are white-background grayscale; calibration is `value = (255 - pixel) / 255 * scale`
- Three data types: RADAR (dBZ, scale=50), RAIN (mm/h, scale=35), PWV (mm, scale=80)
- Server scripts auto-detect `RADAR_2025_S`, `RAIN_2025_S`, `PWV_2025_S` under `DATA_ROOT`
- The `--invert` flag (default: invert) maps white→0; use `--no_invert` when brighter pixels mean stronger values
- Image dimensions must be multiples of 32 (the architecture constraint)

### Shared Utilities (`experiments/common.py`)

Training and test scripts share plumbing via `nowcasting/experiments/common.py`:
- `make_png_dataloader(args, split)` — factory for train/val/test DataLoaders
- `build_generator(args)` / `load_generator_weights(model, path, device)` — model creation and weight loading
- `save_json_args(args, folder)` / `safe_torch_save(obj, path)` — atomic checkpoint saving
- `seed_everything(seed)` — reproducibility

### Key Output Files

From training: `best.ckpt`, `best_state_dict.ckpt`, `latest.ckpt`, `train_log.csv`, `train_log.png`
From testing: `metrics.json` (MAE, RMSE, CSI, POD, FAR, HSS at configurable thresholds), `gt_*.png`/`pd_*.png` frames, `c_*.png` (coupling), `s_*.png` (support gate, V3+), `pwv_*.png` (PWV fields)
From reports: comparison plots (lead-time MAE/RMSE, threshold CSI, neighborhood CSI, power spectrum density)

## Coding Conventions

- snake_case for functions, variables, files; PascalCase for classes
- 4-space indentation
- Keep CLI arguments explicit; use `--model_name` to select registered models rather than importing model classes directly in scripts
- Model variants add files under `nowcasting/models/`; register in `registry.py`
- Shell scripts set environment variables (`DATA_ROOT`, `RUN_ROOT`, `BATCH_SIZE`, `EPOCHS`, `NUM_WORKERS`) consumed by the Python scripts
- Checkpoint files, datasets, and results are gitignored — never commit them
