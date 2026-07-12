#!/usr/bin/env bash
set -e

python -u train_adversarial_custom.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --save_dir ../checkpoints/quick_6h_radar \
    --readme_ckpt ../checkpoints/quick_6h_radar.ckpt \
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
    --log_interval 8 \
    --amp
