#!/usr/bin/env bash
set -e

python -u train_adversarial_custom.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --save_dir ../checkpoints/quick_exp \
    --readme_ckpt ../checkpoints/quick_model.ckpt \
    --device cuda:0 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 2 \
    --epochs 2 \
    --num_workers 2 \
    --max_train_samples 64 \
    --max_val_samples 16 \
    --disc_channels 16 \
    --intensity_scale 128 \
    --pixel_min 0 \
    --pixel_max 255 \
    --log_interval 16 \
    --amp
