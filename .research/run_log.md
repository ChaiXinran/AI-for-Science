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

## 2026-07-23 - First server contrastive-trigger result

- Completed seed 2026 on 2,048 training and 512 matched validation windows.
- Radar CSI10/CSI20 were 0.32569/0.20562; real PWV reached
  0.32645/0.20719, positive deltas of 0.00076/0.00157.
- Real PWV also narrowly exceeded null and temporally reversed PWV at both
  thresholds, while null PWV reproduced the radar metrics as intended.
- The improvement traded higher POD for higher FAR and slightly worsened MAE/
  RMSE. Because temporal reversal preserved most of the gain and only one seed
  is available, the experiment was routed to same-budget multi-seed
  replication rather than full-data training.

## 2026-07-23 - Three-seed replication completed

- All seeds evaluated the same 512 windows with identical sample SHA-256.
- Mean real-minus-radar CSI deltas were +0.00030/+0.00071 at 10/20 mm/h, but
  the effect was smaller than its seed standard deviation and CSI10 changed
  sign in seed 2028.
- Temporal reversal matched or beat real PWV in multiple seed/threshold pairs;
  the mean real advantage over reversal was effectively zero at CSI20.
- MAE worsened for every seed. Both new training curves selected epoch 1, after
  which expanding support and false alarms drove validation weighted-L1 up.
- Marked the positive-only contrastive-trigger residual no-go for full-data
  scaling. The next action is a checkpoint-only input-control diagnostic, not
  another larger training run.

## 2026-07-23 - Input-control leakage fix and diagnostic runner

- Audited the temporal-reverse path and found that reversing all 29 PWV frames
  exposed future PWV in the model's first nine inputs. Invalidated only the old
  temporal-reverse comparisons; real, null, and radar results remain valid.
- Restricted every temporal/spatial diagnostic to the observed prefix and
  tagged outputs with `pwv_control_scope=observed_input_only`.
- Added observed-mean `level_only` and fixed half-domain spatial displacement,
  plus a checkpoint-only three-seed runner and report deltas for both controls.
- Local validation passed: seven model/control tests, Python compilation,
  strict protocol JSON parsing, Bash syntax, and a synthetic multi-seed report.

## 2026-07-23 - Checkpoint-only diagnostics completed

- Completed corrected temporal reversal, observed-period `level_only`, and
  half-domain spatial displacement for seeds 2026--2028 on identical 512-window
  samples.
- Real-minus-level-only CSI was effectively zero (+0.000012/+0.000045 at
  10/20 mm/h), so nearly all final-field behavior survives removal of PWV time
  evolution.
- Spatial displacement roughly halved Birth/Growth PR-AUC, confirming use of
  local spatial alignment, but real-minus-displaced final CSI remained only
  +0.000257/+0.000430 and real MAE was worse.
- Closed the contrastive-trigger mechanism without full-data scaling. Routed
  any successor toward bounded signed calibration with explicit categorical
  rain-threshold supervision and stronger geography/climatology controls.
## 2026-07-23 - Signed PWV calibrator Stage 0 and implementation

- Locked the reviewed chronological split locally: 83 train, 23 validation,
  and 17 test day directories; split SHA256
  `dfc9c425046c04d5f35712f331d7516f941b390d7a3fe7798ef2156ef13816ce`.
- Completed strict Stage-0 audit over 29,511 radar frames and their paired PWV
  frames. Missing pairs: 0. Rejected non-contiguous windows: 0.
- Train-only PWV climatology uses a deterministic frame stride of 6 and has
  audit SHA256
  `d1d5c75952d25a0d69181427589c6981dc5d7ef9836ad88b6e547f0f94e0eb04`.
- Independent 0--2 h positive events:
  - train: 40 / 25 / 8 at 10 / 20 / 30 mm/h;
  - validation: 18 / 16 / 14;
  - test: 13 / 9 / 7.
  Therefore 10 and 20 mm/h are primary; 30 mm/h is diagnostic only.
- PWV level has positive sample-level association with future heavy-rain
  support, while observed first-to-last slope is negative in train and
  validation. Static/anomaly PWV is the primary model; tendency is a secondary
  ablation.
- Implemented `PWVSignedCalibratorNowcastNet`: frozen radar backbone, train-only
  spatial climatology, learned radar gate, exact real-minus-null condition,
  candidate support, and bounded signed source correction.
- Added threshold-balanced 10/20 mm/h trainer, matched spatial-control and
  tendency variants, checkpoint controls, eventwise records, paired day-cluster
  bootstrap report, frozen protocol, and server runner.
- Verification:
  - existing + new unit suite: 9 tests passed;
  - tensor-level null PWV radar identity passed;
  - bounded signed contribution and frozen-backbone gradient tests passed;
  - synthetic end-to-end train/test CLI smoke passed on CPU
    (`SIGNED_CLI_SMOKE_OK`).
- Local NVIDIA hardware is visible, but the `nowcast` environment contains
  CPU-only PyTorch 2.13.0. No environment mutation was made; the scientific
  pilot is assigned to the server GPU.
## 2026-07-23 — Signed calibrator real-data smoke and server checkpoint diagnosis

- The first server attempt completed Stage 0 but stopped before signed-head
  training with 34 incompatible radar-backbone tensors.
- Root cause: the supplied `pwv_birth_growth_v1_radar_gate` checkpoint used
  `input_length=9,total_length=39` (0--3 h), while the signed protocol requires
  `input_length=9,total_length=29` (0--2 h). The output geometry must not be
  force-loaded.
- Updated `run_signed_calibrator_pilot.sh` to retrain the matched 0--2 h radar
  baseline under the same 2048/512 budget before freezing it and training the
  PWV heads.
- Ran a real-data smoke on the local `DATA_2025_S` copy: two train and two
  validation windows, one radar epoch, one signed-head epoch, and real/null
  evaluation.
- The signed model loaded 670 radar tensors with zero missing required tensors;
  the frozen head had 40,674 trainable parameters. Both real and null
  evaluations completed and emitted `metrics.json` plus paired
  `eventwise_records.json`.
- The two-window metrics are pipeline evidence only. They contain no 20 mm/h
  positives and cannot support a CSI claim.
## 2026-07-23 — PWV latent-state fusion implementation and real-data smoke

- Retired `Zero-PWV` as a separate experiment and stopped recursive PWV source
  injection after the signed-source pilot failed its CSI/FAR/MAE gate.
- Implemented `PWVLatentFusionNowcastNet`: separate observed-PWV encoder,
  radar-query/PWV-key-value cross-attention at the 1/8-resolution generative
  latent, learned fusion gate, and future-PWV auxiliary target.
- PWV never enters the recursive radar source or motion equations. Future PWV
  is a training target only and is not consumed by the forward predictor.
- Locked three variants: continued radar-only, aligned PWV, and train-time
  random large spatial displacement. Evaluation uses a deterministic
  half-domain displacement. All begin from the same 0-2 h radar checkpoint.
- Real-data smoke used two train and two validation windows from the local
  `DATA_2025_S`. The latent model loaded 676 radar tensors with zero missing,
  added 19,630 fusion parameters in the light-channel smoke configuration,
  and completed aligned/displaced training plus all three evaluations.
- All three evaluations used sample SHA-256
  `b6a68bf01339077cc4bc71743ed30d6763de375208991728757d47701849bce4`.
  The smoke report completed with null-safe handling of thresholds that contain
  no positive events.
- Ten model/data smoke tests and one paired-bootstrap report test passed.

## 2026-07-23 — Server latent-state pilot

- Completed 2048-train/512-validation, seed 2026, for radar-only, aligned PWV,
  and displaced-PWV variants. All evaluation sample hashes matched.
- Aligned minus radar-only: CSI@10 +0.00342, CSI@20 -0.03199, FAR@10 -0.06275,
  FAR@20 +0.03818, and relative MAE -7.25%.
- Aligned minus displaced: CSI@10 +0.000004, CSI@20 -0.000120, with effectively
  identical MAE. Neither CSI bootstrap interval excluded zero.
- The promotion gate failed. No multi-seed or full-data run is authorized for
  this architecture. The next step is a same-checkpoint aligned-versus-shifted
  inference intervention followed by fusion-residual sensitivity logging.
- Completed the same-checkpoint intervention on all 512 validation windows.
  Aligned-minus-shifted CSI was -0.000078 at 10 mm/h and +0.000026 at
  20 mm/h; both 10,000-repeat day-cluster bootstrap intervals crossed zero.
  Mean paired event MAE changed by 0.000036. This closes the current latent
  fusion model as functionally insensitive to PWV.

## 2026-07-23 — Conditional-information probe implementation

- Added a frozen-radar latent probe with per-lead, 8x8-tile 10/20 mm/h event
  targets for 0-1 h and 1-2 h.
- Radar-only and radar+PWV probes have identical parameter counts (13,378 with
  the server's `ngf=32`; 6,210 in the light-channel smoke). Same-checkpoint PWV
  controls use spatial displacement and guaranteed different-event permutation.
- Added disjoint fit/calibration days, fixed validation thresholds, average
  precision, CSI/POD/FAR, and paired day-cluster bootstrap reporting.
- Four unit tests passed. A 4-train/2-validation real-data smoke completed
  caching, both probe fits, all four evaluations, and null-safe reporting.
  Its metrics are pipeline evidence only.

## 2026-07-23 - Server conditional-information probe

- Completed the 2,048-train/512-validation server run with 1,628 fit and 420
  calibration samples. Both 13,378-parameter probes converged without NaNs;
  final fit losses were 0.067017 (radar-only) and 0.062594 (aligned PWV).
- Aligned PWV minus radar-only CSI:
  - 0--1 h, 10 mm/h: -0.021390, bootstrap 95% CI
    [-0.041234, -0.000155];
  - 0--1 h, 20 mm/h: -0.025629, CI [-0.062962, +0.005709];
  - 1--2 h, 10 mm/h: -0.000020, CI [-0.013197, +0.012099];
  - 1--2 h, 20 mm/h: -0.034686, CI [-0.066808, -0.007011].
- Average precision was lower than radar-only in all four tasks. Aligned PWV
  did beat the spatial-shift control at 1--2 h, especially at 10 mm/h
  (+0.022790 CSI; CI [+0.009542, +0.035437]), but did not consistently beat
  the cross-event control and never beat radar-only.
- Promotion gate: 0/4 tasks passed; safety failed. No multi-seed, full-data, or
  restricted joint-adaptation run is scheduled for this representation.

## 2026-07-23 - Local PWV interpolation audit

- Verified 29,511/29,511 radar PNGs have a same-relative-path PWV PNG. The PWV
  directory contains five additional terminal frames that are not consumed by
  radar windows. Both modalities use the same raw 70x66 raster shape.
- Across 1,181 sampled 30-minute blocks, adjacent six-minute PWV fields had
  mean pixelwise correlation 0.99947 and mean absolute change 0.1234 mm.
  Mean 30-minute endpoint change was only 0.6062 mm.
- Intermediate fields were reconstructable from their 30-minute endpoints
  with mean absolute residual 0.0858 mm. In a separate endpoint check, rounding
  a linear 30-minute interpolation reproduced 80.5% of pixels exactly, versus
  44.1% for a 60-minute interpolation and 29.5% for a 120-minute interpolation.
- Interpretation: the nominal six-minute PWV sequence is predominantly a
  quantized interpolation of a roughly 30-minute product. It should be modeled
  as a slowly varying environmental state, not as nine independent video
  observations. The next information probe should use native-cadence anchors
  over a longer preconditioning window before any new fusion architecture.

## 2026-07-23 - Causal PWV preconditioning probe implementation

- Added a causal three-hour PWV reader using seven 30-minute anchors. Anchors
  are aligned to midnight cadence and the latest anchor is never later than the
  final observed radar timestamp. This avoids possible future-endpoint leakage
  from the six-minute interpolated PWV images.
- Added parameter-matched radar-only, old short-interpolated-PWV, and
  causal-long-PWV probes. The long checkpoint is also evaluated with spatial
  shift, guaranteed cross-event replacement, and tendency-sign reversal.
- Added input-only evaluation strata: all tiles, weak-echo/nondecreasing tiles
  (primary), and radar-quiet tiles (diagnostic). No stratum uses future radar or
  the prediction target to select candidates.
- Six conditional-probe unit tests and the full 16-test model/data smoke suite
  passed. A 4-train/2-validation real-data CLI smoke completed caching, three
  fits, calibration, six variants, stratified metrics, and paired bootstrap.
- A separate 512-window validation support audit found 885--2,955 positive
  tile-leads in the primary weak-echo/nondecreasing stratum across the four
  horizon-threshold tasks, spanning 16--17 positive days. The primary stratum
  therefore has adequate support for the 2,048/512 server pilot.

## 2026-07-23 - Server causal PWV preconditioning probe

- Completed 2,048 training and 512 validation windows with 1,628 fit and 420
  calibration samples. Radar, short-PWV, and causal-long-PWV probes each had
  13,378 parameters over the same 35,035,965-parameter frozen radar model.
- All-window causal-long minus radar-only CSI was +0.01011, +0.06652,
  +0.00190, and +0.03843 for 0--1 h at 10/20 mm/h and 1--2 h at 10/20 mm/h.
  Day-cluster 95% intervals excluded zero for both 0--1 h tasks and 1--2 h at
  20 mm/h. Average precision improved only for 0--1 h at 20 mm/h.
- The primary weak-echo/nondecreasing stratum failed the locked gate (0/4).
  Although CSI exceeded radar-only in three tasks, AP was lower in all four.
- Causal-long PWV did not consistently beat cross-event PWV. All-window CSI
  deltas versus cross-event were +0.00167, -0.01045, +0.00389, and -0.00406;
  the cross-event control was significantly better for 1--2 h at 20 mm/h.
  Tendency reversal changed CSI by at most 0.00593.
- Interpretation: the causal-long branch can improve a calibrated CSI operating
  point relative to radar-only, but the gain cannot be attributed to
  event-specific aligned PWV. Spatial shift sensitivity plus cross-event
  insensitivity points to a stationary geographic/climatological template or
  modality-branch regularization rather than dynamic moisture preconditioning.
- The outer `tee` could race the runner's creation of `PROBE_ROOT`, so no
  `run.log` was retained. The summary is complete; documentation now creates
  `PROBE_ROOT` before starting the pipeline.

## 2026-07-23 - Same-checkpoint PWV attribution implementation

- Added an inference-only decomposition of the causal PWV history into fit-day
  static climatology, a per-anchor valid-domain scalar moisture departure, and
  a zero-spatial-mean event anomaly. Zero padding is excluded from all spatial
  statistics.
- Added five same-checkpoint interventions: real PWV, static climatology,
  static plus event scalar, event scalar without geography, and event spatial
  anomaly without geography or scalar regime. All reuse the trained long-PWV
  checkpoint and its frozen calibration thresholds.
- Added a reconstruction assertion requiring the attribution's real-PWV
  features to exactly match the features used by the trained probe.
- Seven focused unit tests and an inference-only real-cache smoke passed. The
  local smoke reconstructed real features with maximum absolute error 0.0 and
  emitted strict JSON plus paired day-cluster bootstrap results.

## 2026-07-23 - Server same-checkpoint PWV attribution

- Evaluated 512 validation windows through the same 13,378-parameter long-PWV
  checkpoint and frozen thresholds. Real-feature reconstruction error was
  exactly 0.0.
- Static climatology alone produced all-window CSI 0.52919, 0.37890, 0.23053,
  and 0.10808. Relative to the prior radar-only probe, this accounts for about
  63%, 82%, and 86% of the real-PWV CSI gain in 0--1 h/10, 0--1 h/20, and
  1--2 h/20, respectively. It did not explain the small 1--2 h/10 gain.
- Real PWV minus static-climatology-plus-event-scalar CSI was -0.00145,
  +0.01125, +0.00541, and +0.00063 across the four tasks. AP deltas were
  -0.00522, +0.00565, +0.00202, and +0.00337. Only 2/4 point-estimate tasks
  passed, and every day-cluster CSI interval included zero.
- Adding the event scalar to static climatology did not produce a robust
  improvement over static climatology. The spatial-anomaly gate failed 2/4.
- Verdict: most apparent high-threshold CSI improvement is attributable to a
  stationary geographic/climatological template available through PWV, not to
  robust event-specific moisture information.

## 2026-07-23 - Fair dynamic-residual control implementation

- Added the final matched trained comparison. Both probes receive one identical
  fit-day static PWV climatology channel. The dynamic probe alone additionally
  receives latest/mean/scalar-tendency moisture departures and latest/tendency
  zero-spatial-mean event anomalies.
- Static and dynamic probes instantiate the same architecture and initial
  weights and use the same fit/calibration days, class weights, optimizer, and
  epoch budget.
- Added same-checkpoint cross-event, anomaly-only spatial shift,
  tendency-reversal, and static-only interventions. The static climatology is
  never shifted or replaced in these controls.
- Eight focused tests and a 4-train/2-validation real-cache end-to-end smoke
  passed. The smoke emitted matched checkpoints, independent calibration,
  stratified metrics, and paired bootstrap comparisons.

## 2026-07-23 - Server fair dynamic-residual verdict

- Completed the locked 2,048-train/512-validation comparison. Static and
  dynamic probes each had 13,378 parameters and shared identical static
  climatology, radar cache, initialization, fit/calibration days, and training
  budget.
- Dynamic minus separately trained static-climatology CSI was -0.01771,
  -0.01688, +0.00042, and +0.00089 across 0--1 h/10, 0--1 h/20, 1--2 h/10,
  and 1--2 h/20. AP deltas were -0.02056, -0.01518, +0.01237, and +0.00618.
- The 0--1 h/10 CSI day-cluster interval was entirely negative
  [-0.02718, -0.00549]. Both later-horizon CSI deltas were below the locked
  +0.003 minimum and their intervals crossed zero.
- Aligned dynamic PWV was significantly worse than cross-event PWV at
  0--1 h/10 (CSI delta -0.00424; interval [-0.00737, -0.00076]). It did not
  consistently beat spatially shifted anomalies or reversed tendencies.
- Final gate: 0/4 tasks passed; safety failed. Dynamic PWV architecture search
  on the current interpolated product is closed.
