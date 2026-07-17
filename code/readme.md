# NowcastNet with PWV Coupling

Precipitation nowcasting using the NowcastNet architecture with optional
Precipitable Water Vapor (PWV) coupling for false-alarm control.

## Project Structure

```
code/
├── train/                    # Training scripts
│   ├── radar.py              # Radar-only adversarial training
│   └── pwv.py                # PWV model training (main entry)
├── test/                     # Test / evaluation scripts
│   ├── radar.py              # Radar model testing + metric library
│   └── pwv.py                # PWV model testing
├── report/                   # Report generation
│   ├── compare.py            # Multi-experiment comparison report
│   ├── pairwise.py           # A/B pairwise model comparison
│   └── recompute.py          # Recompute metrics from saved PNG samples
├── scripts/                  # Shell script templates
│   ├── env.sh                # Server environment helpers
│   ├── run_all.sh            # Full train+test+report pipeline
│   ├── train_radar.sh        # Server radar training
│   ├── test_radar.sh         # Server radar testing
│   ├── quick_train_radar.sh  # Local quick radar training
│   ├── quick_test_radar.sh   # Local quick radar testing
│   └── quick_test_mrms.sh    # Test with MRMS pretrained weights
├── nowcasting/               # Core library
└── run.py                    # Legacy inference entry
```

## Available Models

| `--model_name` | Script | Description |
|----------------|--------|-------------|
| `NowcastNet` | `train/radar.py` | Original radar-only model |
| `PWVCoupledNowcastNet` | `train/pwv.py` | PWV-coupled with false-alarm control (default) |
| `PWVCoupledNowcastNetObject` | `train/pwv.py` | PWV + auxiliary convective-object head |

## Quick Start (Smoke Test)

```bash
conda activate nowcast
cd code/

# Radar-only (8 samples, 1 epoch, GPU)
python -u train/radar.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --save_dir ../checkpoints/smoke_radar \
    --readme_ckpt ../checkpoints/smoke_radar_model.ckpt \
    --device cuda:0 --img_height 96 --img_width 96 \
    --batch_size 1 --epochs 1 --num_workers 0 \
    --max_train_samples 8 --max_val_samples 2 --disc_channels 8 --amp

# PWV model (CNN source, default)
python -u train/pwv.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --save_dir ../checkpoints/smoke_pwv \
    --readme_ckpt ../checkpoints/smoke_pwv_model.ckpt \
    --device cuda:0 --img_height 96 --img_width 96 \
    --batch_size 1 --epochs 1 --num_workers 0 \
    --max_train_samples 8 --max_val_samples 2 --disc_channels 8

# PWV with cross-attention source
python -u train/pwv.py \
    --pwv_source_type attention \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --save_dir ../checkpoints/smoke_pwv_attn \
    --readme_ckpt ../checkpoints/smoke_pwv_attn_model.ckpt \
    --device cuda:0 --img_height 96 --img_width 96 \
    --batch_size 1 --epochs 1 --num_workers 0 \
    --max_train_samples 8 --max_val_samples 2 --disc_channels 8
```

## Pixel Calibration

Images are white-background grayscale: `value = (255 - pixel) / 255 * scale`.
Use `--no_invert` if brighter pixels mean stronger values.

| Field | Flag | Scale | Notes |
|-------|------|-------|-------|
| RADAR (dBZ) | `--intensity_scale` | 50 | |
| RAIN (mm/h) | `--intensity_scale` | 35 | Use for paper-style verification |
| PWV (mm) | `--pwv_intensity_scale` | 80 | Add `--pwv_invert` |

For rain-rate evaluation (RAIN_2025_S as target):
```bash
--intensity_scale 35 --pixel_min 0 --pixel_max 255
--metric_thresholds 0.5,2,5,10,30
--neighborhood_metric_thresholds 16,32 --neighborhood_size 5
```

## Training

### train/radar.py — Radar-only adversarial training

```bash
python -u train/radar.py \
    --data_root <radar_path> \
    --save_dir <checkpoint_dir> \
    --readme_ckpt <output_model.ckpt> \
    --device cuda:0 \
    --total_length 39 --epochs 60 --batch_size 8
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--forecast_loss` | `weighted_l1` | `weighted_l1` or `facl` |
| `--facl_alpha` | `0.1` | FACL FAL-only fraction |
| `--amp` | off | Enable mixed precision |
| `--init_generator` | `""` | Path to pretrained weights |
| `--resume` | `""` | Resume from checkpoint |

### train/pwv.py — PWV-coupled training (main entry)

All radar flags plus PWV-specific flags below.

**PWV source type:**

| Flag | Default | Description |
|------|---------|-------------|
| `--pwv_source_type` | `cnn` | `cnn` = LightweightUNet, `attention` = temporal cross-attention |

Attention params: `--pwv_attn_dim 64 --pwv_attn_heads 4 --pwv_attn_downsample 4`

**PWV feature engineering:**

| Flag | Default | Description |
|------|---------|-------------|
| `--pwv_tendency_windows` | `""` | Comma-separated minutes, e.g. `"30,60"` |
| `--pwv_tendency_mode` | `slope` | `diff`, `slope`, or `both` |
| `--frame_minutes` | `6` | Minutes per frame |

**PWV loss weights:**

| Flag | Default | Role |
|------|---------|------|
| `--lambda_coupling_smooth` | `0.02` | Spatial smoothness of coupling field |
| `--lambda_coupling_l1` | `0.0005` | Coupling field sparsity |
| `--lambda_align` | `0.05` | Align coupling to rain growth x PWV signal |
| `--lambda_shuffle` | `0.05` | PWV shuffle contrast (0 = disable) |
| `--shuffle_margin` | `0.02` | Margin for shuffle contrast |

**False-alarm control (V3):**

| Flag | Default | Role |
|------|---------|------|
| `--lambda_false_alarm` | `0.25` | Penalize predicted rain in dry/unsupported areas |
| `--lambda_support_dry` | `0.05` | Penalize open support gate in dry areas |
| `--lambda_support_l1` | `0.01` | Support gate sparsity |
| `--false_alarm_threshold` | `2.0` | mm/h below which is "dry" |

**Object head** (`--model_name PWVCoupledNowcastNetObject`):

| Flag | Default | Role |
|------|---------|------|
| `--lambda_object_center` | `0.0` | Set >0 to enable (e.g. `0.5`) |
| `--lambda_object_mask` | `0.0` | Set >0 to enable (e.g. `0.5`) |
| `--lambda_object_dice` | `0.0` | Set >0 to enable (e.g. `1.0`) |
| `--lambda_object_consistency` | `0.0` | Set >0 to enable (e.g. `0.1`) |
| `--object_loss_threshold` | `16.0` | mm/h for object detection |
| `--object_loss_min_area` | `4` | Min pixel area for an object |
| `--object_center_sigma` | `2.0` | Gaussian sigma for center heatmap |
| `--object_consistency_temperature` | `2.0` | Temperature for rain-object Dice |

## Recommended Combinations

**Radar-only:**
```bash
python -u train/radar.py --data_root <radar> --save_dir <dir> --readme_ckpt <ckpt> --device cuda:0
```

**PWV CNN (standard):**
```bash
python -u train/pwv.py \
    --data_root <radar> --pwv_root <pwv> \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --total_length 39 --epochs 60 --batch_size 8
```

**PWV Attention:**
```bash
python -u train/pwv.py --pwv_source_type attention \
    --data_root <radar> --pwv_root <pwv> \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --total_length 39 --epochs 60 --batch_size 8
```

**PWV + Tendency + FACL:**
```bash
python -u train/pwv.py \
    --pwv_tendency_windows "30,60" --pwv_tendency_mode slope \
    --forecast_loss facl --facl_alpha 0.1 \
    --data_root <radar> --pwv_root <pwv> \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --total_length 39 --epochs 60 --batch_size 8
```

**PWV + Object head (init from PWV checkpoint):**
```bash
python -u train/pwv.py --model_name PWVCoupledNowcastNetObject \
    --init_generator ../checkpoints/pwv_model.ckpt \
    --lambda_object_center 0.5 --lambda_object_mask 0.5 \
    --lambda_object_dice 1.0 --lambda_object_consistency 0.1 \
    --data_root <radar> --pwv_root <pwv> \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert \
    --total_length 39 --epochs 40 --batch_size 8
```

## Testing

### test/radar.py — Radar model evaluation

```bash
python -u test/radar.py \
    --checkpoint <model.ckpt> \
    --data_root <radar_path> \
    --output_dir ../results/radar_test \
    --device cuda:0 \
    --split test --max_samples 100 \
    --intensity_scale 35
```

### test/pwv.py — PWV model evaluation

```bash
python -u test/pwv.py \
    --checkpoint <model.ckpt> \
    --data_root <radar_path> \
    --pwv_root <pwv_path> \
    --output_dir ../results/pwv_test \
    --device cuda:0 \
    --split test \
    --intensity_scale 35 --pwv_intensity_scale 80 --pwv_invert
```

Key flags for both:

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | `test` | `train`, `val`, `test`, or `all` |
| `--max_samples` | `0` | Limit samples (0 = all) |
| `--num_save_samples` | `10` | How many to save as PNG |
| `--metric_thresholds` | `"1,5,10,20,40"` | CSI/POD/FAR thresholds (mm/h) |
| `--neighborhood_metric_thresholds` | `""` | Neighborhood CSI thresholds |
| `--neighborhood_size` | `5` | Neighborhood size in pixels |
| `--grid_km` | `1` | Pixel resolution in km |

**Outputs per sample:** `input_*.png`, `gt_*.png`, `pd_*.png`, `ps_*.png`
(persistence baseline). With PWV: `pwv_*.png`, `c_*.png` (coupling field),
`s_*.png` (support gate), `a_*.png` (attention map). With object head:
`oc_*.png` / `om_*.png`. Plus `metrics.json` with full scalar, event,
neighborhood, PSD, CRA, FSS, and object metrics.

## Reports

### report/compare.py — Multi-experiment comparison

Reads `metrics.json` from `run_root/results/*/` for the experiments listed
in the script's EXPERIMENTS dict (Radar-only, PWV, PWV Object). Generates
all comparison plots.

```bash
python -u report/compare.py --run_root <path> --output_dir <path>
```

### report/pairwise.py — A/B model comparison

Compare any two result folders (baseline vs new) with delta summaries.

```bash
python -u report/pairwise.py \
    --baseline_run_root <path1> --baseline_result_dir pwv_3h --baseline_label "CNN" \
    --new_run_root <path2> --new_result_dir pwv_3h --new_label "Attention" \
    --output_dir <path>
```

### report/recompute.py — Recompute metrics from saved PNGs

Re-evaluates saved sample PNGs, useful for recomputing metrics with different
thresholds without re-running the model.

```bash
python -u report/recompute.py --runs_root <path> --output_dir <path> \
    --intensity_scale 35 --device cuda:0
```

## Server Pipeline

```bash
DATA_ROOT=/path/to/DATA_2025_S \
    RUN_ROOT=/path/to/output \
    BATCH_SIZE=8 EPOCHS=60 NUM_WORKERS=8 \
    bash scripts/run_all.sh
```

This runs radar training + testing, then PWV training + testing, then the
comparison report. Customize by editing `scripts/run_all.sh`.

## Quick Shell Scripts

```bash
# Quick adversarial training (local)
bash scripts/quick_train_radar.sh

# Quick radar test (local checkpoint)
bash scripts/quick_test_radar.sh

# Test with MRMS pretrained weights
bash scripts/quick_test_mrms.sh
```

## Data Layout

```
DATA_ROOT/
  RADAR_2025_S/       (or RAIN_2025_S as precipitation proxy)
    202505/
      20250501/
        2025-05-01-01-00-00.png
        2025-05-01-01-06-00.png
        ...
  PWV_2025_S/         (same directory structure as RADAR)
```

Radar and PWV frames are paired by relative path. If a PWV file is missing,
it falls back to zeros. Each sample is a sliding window of `total_length`
consecutive frames (9 input + N output).
