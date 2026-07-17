# NowcastNet with PWV Coupling

Precipitation nowcasting using a two-stage evolution-generative deep learning
model, extended with Precipitable Water Vapor (PWV) coupling for
false-alarm control.

Based on *"Skillful nowcasting of extreme precipitation with NowcastNet"*
(Zhang et al., Tsinghua University / UC Berkeley / CMA, 2023). The original
repository contained inference-only code; this fork adds full training
pipelines, PWV-coupled model variants, and comprehensive evaluation tooling.

## Architecture

```
Input: 9 radar frames (54 min history) + optional 9 PWV frames
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   Evolution Network              PWV Feature Engineering
   (U-Net → motion +              (value, anomaly, delta,
    intensity residuals)           gradient, tendency)
          │                               │
          ▼                               ▼
   Autoregressive Advection        PWV Source Network
   (warp + intensity, 20 steps)    (CNN or Cross-Attention)
          │                               │
          └───────────┬───────────────────┘
                      ▼
              Source Decomposition
       s_t = s_radar + C_s * S_pwv * s_pwv
                      │
                      ▼
              SPADE Generative Decoder
              (noise-injected refinement)
                      │
                      ▼
          Output: 20 precipitation frames (2h forecast)
```

**Key innovation over the original:** PWV (atmospheric moisture) is used as an
auxiliary input to suppress false alarms. A learnable support gate `S_pwv`
prevents PWV from amplifying precipitation predictions in dry regions where
moisture does not support convection.

## Repository Structure

```
├── code/                      # All source code
│   ├── nowcasting/            # Core library
│   │   ├── layers/            # Evolution + Generation network modules
│   │   ├── models/            # NowcastNet, PWV-coupled, Object head
│   │   ├── data_provider/     # PngSequenceDataset (sliding-window PNG loader)
│   │   └── experiments/       # Shared utilities
│   ├── train/                 # Training scripts (radar.py, pwv.py)
│   ├── test/                  # Evaluation scripts (radar.py, pwv.py)
│   ├── report/                # Report generation (compare, pairwise, recompute)
│   ├── scripts/               # Shell script templates
│   └── README.md              # Detailed usage guide
├── environment/               # Docker + conda environment
├── metadata/                  # Code Ocean capsule metadata
├── CLAUDE.md                  # AI assistant guidance
└── README.md                  # This file
```

## Quick Start

```bash
conda activate nowcast
cd code/

# Smoke test: PWV model (8 samples, 1 epoch)
python -u train/pwv.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --save_dir ../checkpoints/smoke_pwv \
    --readme_ckpt ../checkpoints/smoke_pwv_model.ckpt \
    --device cuda:0 --img_height 96 --img_width 96 \
    --batch_size 1 --epochs 1 --num_workers 0 \
    --max_train_samples 8 --max_val_samples 2 --disc_channels 8

# Test
python -u test/pwv.py \
    --checkpoint ../checkpoints/smoke_pwv_model.ckpt \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --output_dir ../results/smoke_pwv_test \
    --device cuda:0 --max_samples 10
```

**Full documentation** — all parameters, model variants, recommended
configurations, report generation, server pipeline — is in [code/README.md](code/README.md).

## Available Models

| Model | Description |
|-------|-------------|
| `NowcastNet` | Original radar-only model |
| `PWVCoupledNowcastNet` | PWV-coupled with false-alarm control (default) |
| `PWVCoupledNowcastNetObject` | PWV + auxiliary convective-object prediction head |

PWV models support two source types: `cnn` (LightweightUNet) or `attention`
(temporal cross-attention, `--pwv_source_type attention`).

## Data Format

Chronological PNG frames organized by day:

```text
data/DATA_2025_S/
  RADAR_2025_S/        (66x70 px, white-background grayscale)
    202505/20250501/2025-05-01-01-00-00.png ...
  PWV_2025_S/          (same structure, same filenames)
  RAIN_2025_S/         (optional — used as precipitation target)
```

Pixel calibration: `value = (255 - pixel) / 255 * scale`.
Radar scale = 50 dBZ, Rain scale = 35 mm/h, PWV scale = 80 mm.

## Environment

```bash
conda create -n nowcast python=3.9
conda activate nowcast
pip install torch torchvision numpy opencv-python scikit-image matplotlib pillow
```

Or use the Dockerfile in `environment/` for a CUDA 11.7 container with all
dependencies pre-installed.

## Citation

```bibtex
@article{zhang2023nowcastnet,
  title   = {Skillful nowcasting of extreme precipitation with {NowcastNet}},
  author  = {Zhang, Yuchen and Long, Mingsheng and Chen, Kaiyuan and
             Xing, Lanxiang and Jin, Ronghua and Jordan, Michael I. and
             Wang, Jianmin},
  journal = {Nature},
  volume  = {619},
  pages   = {526--532},
  year    = {2023}
}
```
