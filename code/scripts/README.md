# Script Layout

Shell workflows are grouped by purpose:

- `mrms/`: MRMS reproduction and local MRMS-style checks.
- `custom/`: local custom-radar train/test entry points.
- `server/`: server-scale train/test/report workflows and shared environment helpers.

The public MRMS reproduction wrappers remain in `code/`, so commands such as
`bash ./mrms_case_test.sh` continue to work from `code/`. Custom and server
workflows should be launched from their grouped script paths, for example
`bash ./scripts/server/server_run_all_3h.sh`.

The scripts under this directory locate the repository's `code/` directory before
running Python commands, so they can be launched either through the wrappers or by
calling the grouped scripts directly.

Server workflows use these defaults unless overridden:

- `DATA_ROOT=/root/autodl-tmp/datasets/north_china/DATA_2025_S`
- `RUNS_ROOT=/root/autodl-tmp/nowcastnet_runs`
- `RUN_ROOT=${RUNS_ROOT}/<version-name>`

Run a version end-to-end:

```bash
bash ./scripts/server/server_run_pwv_v1_3h.sh
bash ./scripts/server/server_run_pwv_v2_3h.sh
bash ./scripts/server/server_run_pwv_v3_3h.sh
bash ./scripts/server/server_run_pwv_v4_3h.sh
bash ./scripts/server/server_run_pwv_v4_tendency_3h.sh
```

Run separate stages:

```bash
bash ./scripts/server/server_train_pwv_v3_3h.sh
bash ./scripts/server/server_test_pwv_v3_3h.sh
bash ./scripts/server/server_report_pwv_v3_3h.sh
```

Original V4 and the V4 tendency variant are separate. The tendency variant
defaults to `PWV_TENDENCY_WINDOWS=30,60`, `PWV_TENDENCY_MODE=slope`, and
`FRAME_MINUTES=6`, and writes to `${RUNS_ROOT}/pwv_v4_tendency_3h`.

Override paths when needed:

```bash
DATA_ROOT=/path/to/DATA_2025_S RUNS_ROOT=/root/autodl-tmp/nowcastnet_runs bash ./scripts/server/server_run_pwv_v3_3h.sh
```
