# Research workspace run log

## 2026-07-22 — Context compression and remote experiment inventory

- Inspected the local repository, recent Git history, training/testing/report entrypoints, and existing `.research/` topic dossier.
- Inventoried the public Hugging Face archive: 390 nodes, 321 files.
- Downloaded 30 small metadata artifacts (README, training arguments, summaries, recomputation manifests) to `.research/hf_audit_raw/`; model weights were not downloaded.
- Found that cross-family comparisons mix `RADAR_2025_S` and `RAIN_2025_S` targets, and unified recomputation covers only 24 saved samples for most models and 10 for one FACL run.
- Wrote an experiment matrix that treats all existing rankings as diagnostic rather than publication-grade.

## 2026-07-22 — Frozen baseline and PWV Birth/Growth implementation

- Added checked split manifests, strict 6-minute continuity, strict PWV pairing, sample IDs, and SHA-256 provenance.
- Added `PWVBirthGrowthNowcastNet` with explicit birth probability, growth probability, non-negative source amount, and direct supervision relative to a frozen radar evolution baseline.
- Added matched-seed server orchestration, a smoke mode, specialized birth/growth metrics, and comparison hash checks.
- Initial local validation covered Python AST/compile checks and a synthetic split/pairing test; CUDA runtime validation was added in the following entry.

## 2026-07-22 — Local CUDA smoke validation

- The requested `nowcast` environment contains CPU-only PyTorch 2.13; used the existing `aipdr` environment with PyTorch 2.8.0+cu126 on the local RTX 4060 without modifying either environment.
- Added repeatable unit smoke tests for strict data pairing/provenance, PWV controls, Birth/Growth losses and metrics, frozen-backbone gradients, radar-checkpoint mapping, and the exact 9-input/30-output 96x96 tensor contract.
- Found and fixed a hard-coded noise-projector reshape that was incompatible with the frozen 96x96 protocol; both radar and PWV models now reshape from the actual noise-grid dimensions.
- Five GPU unit tests passed. A real command-line smoke run also completed radar train/test and PWV Birth/Growth train/test on temporary paired synthetic data, producing checkpoints and metrics successfully.
