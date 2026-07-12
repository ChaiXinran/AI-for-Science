# Custom NowcastNet Training Workflow

This project now supports training and testing NowcastNet on the local PNG radar dataset under `data/DATA_2025_S/RADAR_2025_S`.

## Data Layout

The custom loader expects chronological PNG frames grouped by day:

```text
data/DATA_2025_S/RADAR_2025_S/
  202505/
    20250501/
      2025-05-01-01-00-00.png
      2025-05-01-01-06-00.png
      ...
```

Each sample is a sliding 29-frame window: 9 input frames and 20 target frames. Your source images are `66x70`; the loader pads them to `96x96`, which is the smallest practical multiple of 32 for the current NowcastNet architecture.

## Pixel Calibration

By default, images are treated as white-background radar maps:

```text
intensity = (255 - pixel) / 255 * 128
```

This is controlled by:

```bash
--intensity_scale 128 --pixel_min 0 --pixel_max 255
```

Use `--no_invert` only if brighter pixels mean stronger precipitation. For physically meaningful rain-rate evaluation, replace these defaults with the real mapping from pixel values to precipitation intensity.

## GPU Smoke Test

Run this first in WSL:

```bash
conda activate nowcast
cd /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/code

python -u train_adversarial_custom.py \
  --data_root ../data/DATA_2025_S/RADAR_2025_S \
  --save_dir ../checkpoints/smoke_adv_gpu \
  --readme_ckpt ../checkpoints/smoke_mrms_model.ckpt \
  --device cuda:0 \
  --img_height 96 \
  --img_width 96 \
  --batch_size 1 \
  --epochs 1 \
  --num_workers 0 \
  --max_train_samples 8 \
  --max_val_samples 2 \
  --disc_channels 8 \
  --log_interval 1 \
  --amp
```

## Full Training

```bash
cd /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/code
bash ./train_nowcastnet_adversarial.sh
```

Outputs:

```text
checkpoints/custom_nowcastnet_adv/best.ckpt
checkpoints/custom_nowcastnet_adv/best_state_dict.ckpt
checkpoints/custom_nowcastnet_adv/latest.ckpt
checkpoints/custom_nowcastnet_adv/train_log.csv
checkpoints/custom_nowcastnet_adv/train_log.png
checkpoints/mrms_model.ckpt
```

`checkpoints/mrms_model.ckpt` is the README-style generator weight file for inference.

## PWV-Coupled Model

The first research iteration keeps the original radar-only model intact and adds a separate PWV-coupled variant:

```text
s_t = s_t^radar + C_t(x, y) * s_t^PWV
```

where `C_t(x, y)` is a sigmoid-bounded coupling field saved as `c_*.png` during testing. This field is intended to show where current PWV is allowed to contribute to precipitation growth or maintenance.

Quick PWV experiment:

```bash
cd /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/code
bash ./quick_train_pwv.sh
bash ./quick_test_pwv.sh
```

Outputs:

```text
checkpoints/quick_pwv_coupled/
checkpoints/quick_pwv_coupled.ckpt
results/quick_pwv_coupled/sample_0000/c_*.png
results/quick_pwv_coupled/sample_0000/pwv_*.png
```

Use this against the radar-only quick experiment:

```bash
bash ./quick_train.sh
bash ./quick_test.sh
```

The intended comparison order is:

```text
Persistence baseline
Radar-only NowcastNet
PWV-coupled NowcastNet
```

## PWV-Coupled V2

V2 is the recommended next research branch for making PWV physically useful. It keeps the original radar-only model and the first PWV model unchanged, then adds:

```text
PWV features = raw value + anomaly + temporal change + spatial gradient
s_t = s_t^radar + C_t(x, y) * s_t^PWV
```

Compared with the first PWV version, `C_t(x, y)` is predicted by an independent lightweight U-Net instead of the zero-gated evolution network. Training also adds a weak physical alignment loss that encourages high coupling where future rain grows and PWV has growth or gradients, plus a PWV-shuffle contrast term.

Three-hour V2 quick experiment:

```bash
cd /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/code
bash ./quick_train_pwv_v2_3h.sh
bash ./quick_test_pwv_v2_3h.sh
```

The focused V2 variant is mainly an ablation for testing strong physical constraints. It initializes from the 3-hour Radar-only checkpoint if available, freezes the radar backbone for the first epoch, and emphasizes the `1-3h` range, precipitation growth, and high-intensity tails:

```bash
bash ./quick_train_pwv_v2_focus_3h.sh
bash ./quick_test_pwv_v2_focus_3h.sh
```

If the focused variant becomes too conservative, use the tuned V2 branch. It keeps the V2 architecture, avoids radar freezing, disables the extreme-tail loss, and applies only light PWV alignment and growth guidance:

```bash
bash ./quick_train_pwv_v2_tuned_3h.sh
bash ./quick_test_pwv_v2_tuned_3h.sh
```

Outputs:

```text
checkpoints/quick_3h_pwv_v2/
checkpoints/quick_3h_pwv_v2.ckpt
results/quick_3h_pwv_v2/metrics.json
results/quick_3h_pwv_v2/sample_0000/c_*.png
checkpoints/quick_3h_pwv_v2_focus/
results/quick_3h_pwv_v2_focus/metrics.json
checkpoints/quick_3h_pwv_v2_tuned/
results/quick_3h_pwv_v2_tuned/metrics.json
```

V2 currently trains in full precision by default for numerical stability. GPU is still used through `--device cuda:0`; mixed precision is deliberately disabled inside the V2 trainer for now.

## 3-6 Hour Experiments

The default quick models predict 20 future frames, about 2 hours for 6-minute data. Longer horizons require new checkpoints because the model output channel count changes.

Three-hour quick comparison:

```bash
cd /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/code
bash ./quick_train_3h.sh
bash ./quick_test_3h.sh
bash ./quick_train_pwv_3h.sh
bash ./quick_test_pwv_3h.sh
```

Six-hour quick comparison uses fewer samples by default to keep exploration manageable:

```bash
bash ./quick_train_6h.sh
bash ./quick_test_6h.sh
bash ./quick_train_pwv_6h.sh
bash ./quick_test_pwv_6h.sh
```

Long-horizon tests write `lead_time_metrics` for every forecast frame and `horizon_metrics` for bins such as `0-1h`, `1-2h`, `2-3h`, and `3-6h`. After running the tests, generate plots with:

```bash
python code/make_horizon_report.py
```

The report is saved to:

```text
reports/horizon_comparison/
```

## Testing

```bash
cd /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/code
bash ./mrms_custom_case_test.sh
```

Outputs:

```text
results/us/metrics.json
results/us/sample_0000/input_*.png
results/us/sample_0000/gt_*.png
results/us/sample_0000/pd_*.png
```

`metrics.json` includes MAE, MSE, RMSE, and threshold metrics: CSI, POD, FAR, BIAS, and HSS for thresholds `1,5,10,20,40`.

## Notes

This implementation follows the NowcastNet paper structure at an engineering level: generator, temporal discriminator, adversarial loss, evolution accumulation loss, motion regularization, and spatial pooling loss. It is not the original authors' full private training pipeline, because the released capsule only contained inference code.
