#!/usr/bin/env bash
set -e

python -u test_custom.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --checkpoint ../checkpoints/quick_3h_radar.ckpt \
    --output_dir ../results/quick_3h_radar \
    --device cuda:0 \
    --split test \
    --input_length 9 \
    --total_length 39 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 1 \
    --num_workers 2 \
    --max_samples 20 \
    --num_save_samples 10 \
    --metric_thresholds 1,5,10,20,40 \
    --frame_minutes 6 \
    --horizon_bins 0-1,1-2,2-3
