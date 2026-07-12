#!/usr/bin/env bash
set -e

python -u train_adversarial_custom.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --save_dir ../checkpoints/quick_3h_radar \
    --readme_ckpt ../checkpoints/quick_3h_radar.ckpt \
    --device cuda:0 \
    --input_length 9 \
    --total_length 39 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 2 \
    --epochs 2 \
    --num_workers 2 \
    --max_train_samples 64 \
    --max_val_samples 16 \
    --disc_channels 16 \
    --log_interval 16 \
    --amp
