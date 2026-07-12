#!/usr/bin/env bash
set -e

python -u train_pwv_coupled.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --save_dir ../checkpoints/quick_pwv_coupled \
    --readme_ckpt ../checkpoints/quick_pwv_coupled.ckpt \
    --device cuda:0 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 2 \
    --epochs 2 \
    --num_workers 2 \
    --max_train_samples 64 \
    --max_val_samples 16 \
    --disc_channels 16 \
    --evo_base_channels 32 \
    --intensity_scale 128 \
    --pixel_min 0 \
    --pixel_max 255 \
    --pwv_intensity_scale 1 \
    --pwv_pixel_min 0 \
    --pwv_pixel_max 255 \
    --lambda_coupling_smooth 0.02 \
    --lambda_coupling_l1 0.001 \
    --log_interval 16 \
    --amp
