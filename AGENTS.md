# Repository Guidelines

## Project Structure & Module Organization

This repository packages the NowcastNet inference code. The main working tree is `code/`: `run.py` is the command-line entry point, `run` is the Code Ocean launch wrapper, and `requirements.txt` lists Python dependencies. Model code lives in `code/nowcasting/models/`, data loading in `code/nowcasting/data_provider/`, layer implementations in `code/nowcasting/layers/`, and output rendering/evaluation helpers in `code/nowcasting/evaluator.py`. Reproducibility scripts are `code/mrms_case_test.sh` and `code/mrms_large_case_test.sh`. Container setup is kept under `environment/`, while `metadata/` contains capsule metadata.

## Build, Test, and Development Commands

Run commands from `code/` unless noted.

```bash
pip install -r requirements.txt
bash ./mrms_case_test.sh
bash ./mrms_large_case_test.sh
python -u run.py --device cuda:0 --dataset_path /data/dataset/mrms/figure --pretrained_model /data/checkpoints/mrms_model.ckpt --gen_frm_dir /results/us/ --model_name NowcastNet
```

`pip install` prepares the Python environment. The two shell scripts reproduce the normal and large MRMS figure cases. The direct `run.py` form is useful when changing paths, image sizes, sample counts, or device settings. GPU execution is expected for full-size runs.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Follow existing concise module style: snake_case for functions, variables, and files; PascalCase only for classes such as `Model`; constants are rare and should be uppercase if added. Keep CLI arguments explicit in `run.py`, and prefer small, focused modules under the existing `nowcasting` package. Avoid broad rewrites while making model or data-loader changes because tensor shapes and output paths are tightly coupled.

## Testing Guidelines

There is no standalone unit test suite in this capsule. Treat the MRMS scripts as integration checks and verify that `results/.../test_result/` contains generated `gt*.png` and `pd*.png` frames. For smaller edits, run `python -m py_compile` on touched Python files. When changing loading, shapes, or visualization logic, test both `normal` and `large` case paths if data and checkpoint files are available.

## Commit & Pull Request Guidelines

The current history uses short release-style messages, for example `Version 1.0`. Keep commits brief and imperative, such as `Update data loader paths` or `Fix NowcastNet output rendering`. Pull requests should describe the changed code path, list the command or script used for validation, note required datasets/checkpoints, and include representative output screenshots when visualization behavior changes.

## Security & Configuration Tips

Do not commit MRMS datasets, generated results, or checkpoint files. Keep local paths such as `/data/dataset/mrms/figure`, `/data/checkpoints/mrms_model.ckpt`, and `/results/us/` configurable through script arguments rather than hard-coded into library modules.
