#!/usr/bin/env bash

init_server_defaults() {
    local run_name="$1"

    DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/north_china/DATA_2025_S}"
    RUNS_ROOT="${RUNS_ROOT:-/root/autodl-tmp/nowcastnet_runs}"
    RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/${run_name}}"
    DEVICE="${DEVICE:-cuda:0}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
    EPOCHS="${EPOCHS:-60}"
    NUM_WORKERS="${NUM_WORKERS:-8}"

    export DATA_ROOT RUNS_ROOT RUN_ROOT DEVICE BATCH_SIZE TEST_BATCH_SIZE EPOCHS NUM_WORKERS
    mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results" "${RUN_ROOT}/reports"
}

run_server_report() {
    local report_name="$1"
    mkdir -p "${RUN_ROOT}/logs"
    run_python_module nowcasting.cli.reports.server_3h \
        --run_root "${RUN_ROOT}" \
        --output_dir "${RUN_ROOT}/reports/${report_name}" \
        2>&1 | tee "${RUN_ROOT}/logs/report_${report_name}.log"
}
