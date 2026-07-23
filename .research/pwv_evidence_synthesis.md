# PWV evidence synthesis and pre-experiment decision

Date: 2026-07-23

## Executive verdict

Continue the PWV direction, but change the scientific claim and the model role.
The literature provides credible positive evidence that GNSS-derived column
moisture can improve high-intensity precipitation forecasts. However, direct
radar+GNSS-PWV nowcasting has already been published, so a generic multimodal
fusion paper is not a sufficient novelty target.

The best-supported and project-consistent hypothesis is:

> Spatially aligned PWV state improves 0--2 h heavy-rain occurrence skill only
> in radar-identified dynamically plausible or uncertain regions; PWV temporal
> tendency adds value only if it beats a static/anomaly control.

This is different from the rejected hypothesis that PWV should create a dense,
positive birth/growth source.

## Evidence that the direction is viable

1. Liu et al. (IEEE TGRS 2025, DOI 10.1109/TGRS.2025.3554745) directly fuse
   GNSS-PWV and radar over Hong Kong and report a 26% relative CSI improvement
   at 30 mm/h for the complete fusion-plus-attention model.
2. Lu et al. (IEEE TGRS 2025, DOI 10.1109/TGRS.2025.3587883) use radar QPE,
   satellite SWD, and GNSS ZTD for 0--120 min forecasts. Across 189 periods,
   the abstract reports MAE 0.34 mm/h, RMSE 0.61 mm/h, and improvements in
   higher-threshold CSI/FSS.
3. Sun et al. (Remote Sensing 2026, DOI 10.3390/rs18121929) report +1 h
   torrential-rain CSI 0.409 for radar+PWV versus 0.345 for their single-source
   model in Beijing--Tianjin--Hebei.
4. Trentini et al. (arXiv:2607.05658) report that ZWD improves Aurora
   precipitation ETS most strongly at the heavy tail, reaching 8.8% at the
   99th percentile, with paired checkpoint evidence.

These papers establish plausibility, not a guarantee that this dataset or the
current architecture will improve.

## What the literature does not establish cleanly

- Several gains bundle PWV/ZTD with attention, satellite data, DEM/time
  encoding, architecture changes, or a different loss. They do not isolate the
  incremental information in PWV under matched model capacity.
- Relative gains can look large when the baseline CSI is small. Absolute CSI,
  event counts, event-held-out uncertainty, frequency bias, and FAR are often
  incomplete.
- Resampling 30 min PWV to a radar-like 6 min grid does not create genuine
  6 min moisture dynamics.
- PWV-only occurrence models repeatedly attain high POD with high FAR. Moisture
  is a precondition, not an initiation trigger.
- The project's own corrected diagnostics show that `level_only` retains
  nearly all final-field skill, while temporal order adds little. Therefore the
  temporal-tendency claim is currently unsupported.

## CCF-A novelty boundary

The paper should not be framed as “we fuse PWV with radar.” A stronger and more
defensible contribution would combine:

1. a falsifiable physical hypothesis (moisture condition AND radar trigger),
2. a model with exact null identity and bounded signed influence,
3. controls that isolate content, time, space, climatology, and station geometry,
4. event-held-out and cross-season/region validation with paired uncertainty,
5. evidence explaining when PWV helps, not only a point estimate of average CSI.

That package is materially more rigorous than the closest prior art and aligns
with the failure observed in the positive-only residual head.

## Proposed successor architecture

Keep the trained radar forecast as the identity baseline. Freeze it for the
first pilot so the experiment measures added PWV information rather than a
changed radar model.

- Radar branch: predict a dynamic/uncertainty gate from radar latent features,
  recent radar growth, and proximity to the 10/20 mm/h decision boundaries.
- PWV branch: encode absolute level and leakage-safe seasonal anomaly
  separately. Add temporal slope/variation only in a secondary variant.
- Interaction: use a small cross-attention or FiLM-style module; PWV can act
  only through the radar gate.
- Output: a bounded signed residual or signed threshold-logit correction. Zero
  PWV/control input must reproduce radar exactly. Both enhancement and
  suppression are allowed.
- Objective: weighted/focal binary losses at 10 and 20 mm/h, a small
  preservation loss on continuous rain rate, and explicit penalties/selection
  criteria for FAR and frequency bias. Do not optimize a positive source mask.

## Required comparisons

Use identical initialization, windows, optimizer budget, and parameter count.
Distinguish train-time ablations from checkpoint-only stress tests.

1. Radar-only identity baseline.
2. Null-PWV matched head.
3. Real PWV level plus climatological anomaly (primary variant).
4. Event-wise shuffled PWV or fixed spatial displacement (content/location
   negative control; avoid breaking the train/validation split).
5. Real PWV plus temporal tendency (secondary incremental variant).
6. Optional station-distance/coverage mask without PWV values (geometry
   shortcut control).

Checkpoint-only reverse/level/shift tests remain useful, but they cannot replace
matched train-time controls because they may introduce distribution shift.

## Staged experiment gate

### Stage 0: data and metric audit, no training

- Count independent events and positive pixels by horizon at 10, 20, and
  optionally 30 mm/h.
- Verify that entire storms stay in one split.
- Quantify PWV cadence, missingness, station distance, interpolation method,
  and correlation between level/anomaly/tendency.
- Build climatology from training data only.
- Estimate event-bootstrap noise for CSI so the minimum meaningful effect is
  committed before model results are viewed.

### Stage 1: local/small server pilot

Use the existing 2048/512 development manifest, one seed, a frozen radar
checkpoint, and only the small conditional head. The objective is debugging and
effect-size screening, not a paper claim.

### Stage 2: replication gate

Run at least three seeds on the same development protocol and report paired
event-bootstrap intervals. Promote only if real static/anomaly PWV beats radar,
null, and location/content controls without an unacceptable FAR/bias trade.
Temporal tendency is promoted only if it also beats the static/anomaly variant.

### Stage 3: full-data evidence

Only after Stage 2 passes: use event-held-out full data, a locked test set, and
preferably another season or region. Report horizon-wise CSI/POD/FAR/bias,
FSS/object metrics, calibration/Brier score for the trigger, MAE/RMSE, and
spectral diagnostics.

## Immediate next action

Do Stage 0 before writing another model. Its output should be a machine-readable
support audit and a one-page locked protocol containing the primary threshold,
minimum effect, controls, and promotion rule. Then implement only the smallest
frozen-radar signed calibrator needed for Stage 1.

