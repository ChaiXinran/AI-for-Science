# PWV integration redesign for 0-2 h CSI

## Diagnosis of the current model

The current Birth/Growth head computes an OR-like activation:

```text
activation = 1 - (1 - birth_probability) * (1 - growth_probability)
contribution = activation * softplus(amount)
```

Either head can therefore create a positive source. Both heads also receive
radar-derived features, so zero PWV is not an identity path. This conflicts
with the physical evidence that high or rising PWV is a necessary moisture
condition but is insufficient without a convective trigger. It also explains
the observed pattern: high POD, very high FAR, degraded 0-1 h CSI, and no
PWV-attributable 1-2 h CSI gain.

## Recommended direction: contrastive AND-gated object residual

Use the frozen radar model as the default forecast and allow PWV to modify it
only when three conditions agree:

```text
candidate region from radar
AND radar dynamic trigger
AND PWV moisture sufficiency / accumulation
```

The residual should be identity-preserving:

```text
raw_real = H(radar_features, pwv_features)
raw_null = H(radar_features, null_pwv_features)
pwv_evidence = relu(raw_real - raw_null)
delta_source = candidate_mask * trigger_probability * pwv_evidence * amount
```

When PWV is null, `raw_real == raw_null`, so `delta_source == 0` exactly. This
makes radar-only the architectural default rather than merely a separate
baseline.

### PWV representation

Keep the existing delta and spatial-gradient features, but add explicitly:

- physical absolute PWV after correct image-value inversion;
- month/day-of-year climatological percentile or z-score;
- 30 min and 60 min accumulated PWV change;
- 30 min and 60 min robust slope;
- positive accumulation and drying as separate channels;
- local moisture-gradient magnitude and convergence proxy.

Do not normalize away the absolute moisture level within each sample. Encode
absolute/climatological features in one branch and transient changes in a
second branch.

### Radar trigger and candidate support

Construct candidates from radar information that is available at inference:

- existing convective objects and a dilated boundary around them;
- weak-echo regions adjacent to growing cells;
- radar-source uncertainty or disagreement between evolution and generative
  forecasts;
- recent radar intensity tendency and convergence/deformation features.

The PWV module should rank patches or objects inside this support, not classify
every grid pixel. An object/patch probability can be broadcast with a learned
spatial kernel to preserve coherent precipitation shapes.

### Fusion and lead conditioning

- Use separate radar and PWV encoders.
- Fuse encoded tokens with small cross-attention or FiLM, not input-channel
  concatenation.
- Condition scale and bias on the forecast lead, following the Temp-AdaLN idea.
- For 0-2 h, initialize PWV influence near zero at early leads and let validation
  data learn whether it should increase toward 60-120 min. Do not hard-code a
  2-3 h gate because that interval is outside the primary task.

### Training targets and losses

Primary training unit: candidate patch/object, with dense reconstruction as a
secondary constraint.

- class-balanced focal or ranking loss for birth/growth inside candidates;
- soft CSI or Tversky loss at 10 and 20 mm/h;
- neighborhood/object loss for spatial tolerance and coherence;
- explicit false-positive cost stratified by lead time;
- counterfactual margin requiring real PWV to outperform null and
  temporal-reversed PWV on positive candidate events;
- optional small ensemble with CRPS after the deterministic pilot succeeds.

Avoid optimizing global MAE as the dominant PWV objective because dry pixels
overwhelm the rare initiation/growth signal.

## Alternative directions

### B. PWV-conditioned probabilistic adaptor

Keep radar NowcastNet deterministic. Train a conditional residual flow or
diffusion adaptor whose prior is the radar forecast and whose conditioning is
PWV. Optimize CRPS/energy score and threshold reliability. This is closer to
the multimodal and generative-assimilation papers, but is substantially more
expensive and should follow a successful deterministic pilot.

### C. PWV auxiliary representation pretraining

Pretrain a PWV encoder to predict radar-object growth/onset over 30-120 min and
to distinguish correctly aligned from temporally reversed PWV. Freeze or
lightly fine-tune it when conditioning the radar model. This is lower risk than
direct source injection but may yield a smaller headline contribution.

## Minimal falsification-first experiment

Before building a large model:

1. Reclassify the current 512-window evaluation as development evidence.
2. On train/validation only, measure Birth/Growth PR-AUC for absolute PWV,
   climatological anomaly, 30/60 min change, and slope within radar candidate
   regions, separately for 0-1 h and 1-2 h.
3. Train the contrastive AND-gated head on the same 2048/512 pilot budget.
4. Compare radar, null PWV, real PWV, and temporal-reversed PWV with identical
   sample hashes.
5. Promote only if real PWV improves both CSI10 and CSI20 over radar and null
   PWV for pooled 0-2 h, while 0-1 h CSI degradation is negligible and FAR/bias
   remain calibrated.
6. Only then run full data, three seeds, event bootstrap intervals, and a new
   event-held-out final test set.

## Direction ranking

| Direction | Expected CSI potential | Implementation risk | Scientific attribution | Recommendation |
|---|---:|---:|---:|---|
| Contrastive AND-gated object residual | High | Medium | High | Build first |
| PWV auxiliary representation pretraining | Medium | Low-medium | Medium-high | Backup / ablation |
| PWV-conditioned probabilistic adaptor | High | High | Medium | Phase 2 only |
| Continue tuning the current dense positive source | Low | Medium | Low | Stop |

