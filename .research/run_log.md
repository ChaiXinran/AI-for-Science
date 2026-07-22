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

## 2026-07-22 — Real-data prevalence audit and representative smoke

- Audited all 29,511 local `RAIN_2025_S` frames under the reviewed split. Test
  windows containing at least one 10/20 mm/h target pixel are 46.9%/26.9%, but
  unique strong-rain pixel rates are only 0.726%/0.174%.
- The original four-sample smoke selected only the first test windows. Added
  deterministic uniform limited-sample selection and expanded smoke mode to
  64 train, 32 validation, and 32 test windows.
- Replaced non-finite JSON numbers with `null` and enforced strict JSON output.
- A one-epoch real-data CUDA smoke completed for radar, zero-PWV, and real-PWV.
  The radar baseline still had CSI=0 at 10/20 mm/h, so no PWV effectiveness
  conclusion is permitted from this run.
- The radar-relative labels were approximately 0.023% Birth and 0.022% Growth
  in the smoke training subset. Replaced globally averaged focal/source losses
  with separately normalized positive/negative and active/inactive losses.

## 2026-07-22 — Server radar gate

- The 2048-train/512-validation radar pilot trained for 10 epochs without
  instability; validation weighted-L1 improved from 0.918 to a best 0.745 at
  epoch 7.
- On 512 held-out windows, model MAE/RMSE were 0.163/1.096, and CSI at 10/20
  mm/h was 0.239/0.201. All improved over persistence (0.209/1.360 and
  0.169/0.123, respectively).
- Added a matched PWV pilot runner that reuses the passed radar checkpoint and
  compares zero versus real PWV with identical losses and sample identities.

## 2026-07-22 - Matched PWV pilot diagnosis

- The 512-window zero-PWV and real-PWV evaluations used identical sample hashes.
- Both source-head variants underperformed the frozen radar forecast; real PWV
  was worse than zero PWV on MAE, RMSE, CSI10, CSI20, and Birth PR-AUC.
- Added 0--1 h, 1--2 h, and 2--3 h threshold-event metrics to radar and PWV
  evaluators and exposed them in the protocol comparison report.
- Added `RETEST_ONLY=1` to the pilot runner so existing best checkpoints can be
  re-evaluated with the new metrics without retraining.
- Local validation: Python compilation passed, all five Birth/Growth smoke tests
  passed in the `nowcast` environment, and a synthetic horizon-CSI check passed.

## 2026-07-22 - Long-lead PWV signal found

- Re-evaluated the three matched checkpoints on the same 512 windows and split
  CSI10/CSI20 into 0--1 h, 1--2 h, and 2--3 h bins.
- Real PWV was harmful at 0--1 h, mixed at 1--2 h, but improved over both
  zero-PWV and radar-only at 2--3 h: CSI10=0.07023 and CSI20=0.02149.
- Classified the result as a mechanism-positive but architecture-negative
  pilot: preserve the long-lead hypothesis and redesign the injection path.

## 2026-07-22 - Project-horizon correction

- Confirmed that the intended task is 0--2 h, not 0--3 h.
- Reclassified the 2--3 h gain as out-of-primary-scope exploratory evidence.
- Added exact 0--2 h event aggregation to the comparison report by pooling the
  existing 0--1 h and 1--2 h contingency counts; no new inference is required.

## 2026-07-22 - Contrastive-trigger pilot implementation

- Replaced sequential naming for new work with a mechanism name plus an
  executable protocol: `pwv_contrastive_trigger_pilot`.
- Added a radar-trigger AND PWV-evidence residual whose null-PWV contribution
  is exactly zero, together with real, null, and temporal-reversal controls
  evaluated from the same checkpoint.
- Corrected the experiment contract to 9 input frames and 20 forecast frames
  (0--2 h) and required a newly matched radar checkpoint because the archived
  30-frame checkpoint has incompatible output dimensions.
- Added deterministic evaluation noise, exact sample-hash checks, CSI/POD/FAR/
  bias reporting at 10 and 20 mm/h, and a single-command server pilot runner.
- Local CUDA unit coverage passed for 96x96 9-to-20 inference and strict
  tensor-level null-PWV identity. The command-line smoke uses `abs_tol=1e-4`
  and `rel_tol=1e-5` for independent-process aggregates while retaining exact
  event-count checks.
