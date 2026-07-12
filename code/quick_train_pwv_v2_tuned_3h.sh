#!/usr/bin/env bash
set -e

python -u train_pwv_coupled_v2.py \
    --data_root ../data/DATA_2025_S/RADAR_2025_S \
    --pwv_root ../data/DATA_2025_S/PWV_2025_S \
    --save_dir ../checkpoints/quick_3h_pwv_v2_tuned \
    --readme_ckpt ../checkpoints/quick_3h_pwv_v2_tuned.ckpt \
    --device cuda:0 \
    --input_length 9 \
    --total_length 39 \
    --img_height 96 \
    --img_width 96 \
    --batch_size 2 \
    --epochs 3 \
    --num_workers 2 \
    --max_train_samples 128 \
    --max_val_samples 24 \
    --disc_channels 16 \
    --evo_base_channels 32 \
    --pwv_base_channels 24 \
    --lambda_coupling_smooth 0.015 \
    --lambda_coupling_l1 0.0003 \
    --lambda_align 0.015 \
    --lambda_shuffle 0.04 \
    --lambda_growth 0.02 \
    --lambda_extreme 0 \
    --shuffle_margin 0.02 \
    --frame_minutes 6 \
    --lead_focus_start_hour 1 \
    --lead_focus_end_hour 3 \
    --lead_focus_factor 0.25 \
    --log_interval 16
