#!/usr/bin/env bash
set -e

python -u train/radar.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --save_dir ../checkpoints/custom_nowcastnet_adv \
    --readme_ckpt ../checkpoints/mrms_model.ckpt \
    --device cuda:0 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 2 \
    --epochs 50 \
    --lr_g 0.0001 \
    --lr_d 0.0004 \
    --num_workers 2 \
    --stride 1 \
    --intensity_scale 128 \
    --pixel_min 0 \
    --pixel_max 255 \
    --lambda_forecast 1.0 \
    --lambda_evolution 0.5 \
    --lambda_advected 0.25 \
    --lambda_motion 0.02 \
    --lambda_pool 0.2 \
    --lambda_adv 0.01 \
    --log_interval 20 \
    --amp
