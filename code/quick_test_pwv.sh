#!/usr/bin/env bash
set -e

python -u test_pwv_coupled.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --checkpoint ../checkpoints/quick_pwv_coupled.ckpt \
    --output_dir ../results/quick_pwv_coupled \
    --device cuda:0 \
    --split test \
    --img_height 96 \
    --img_width 96 \
    --batch_size 1 \
    --num_workers 2 \
    --max_samples 20 \
    --num_save_samples 10 \
    --intensity_scale 128 \
    --pixel_min 0 \
    --pixel_max 255 \
    --pwv_intensity_scale 1 \
    --pwv_pixel_min 0 \
    --pwv_pixel_max 255 \
    --metric_thresholds 1,5,10,20,40
