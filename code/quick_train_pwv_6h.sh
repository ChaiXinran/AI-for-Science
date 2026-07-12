#!/usr/bin/env bash
set -e

python -u train_pwv_coupled.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --save_dir ../checkpoints/quick_6h_pwv \
    --readme_ckpt ../checkpoints/quick_6h_pwv.ckpt \
    --device cuda:0 \
    --input_length 9 \
    --total_length 69 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 1 \
    --epochs 1 \
    --num_workers 2 \
    --max_train_samples 32 \
    --max_val_samples 8 \
    --disc_channels 8 \
    --evo_base_channels 16 \
    --lambda_coupling_smooth 0.02 \
    --lambda_coupling_l1 0.001 \
    --log_interval 8 \
    --amp
