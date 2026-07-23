---
project: "NowcastNet with PWV Coupling"
last_updated: "2026-07-23"
stage: design
status: draft
source: "topic_dossier.gaps.yml#G1"
gap_verdict: "conditional-go — Proceed only if event-independent pilots show PWV adds robust information"
placeholder_segments: []
---

# Design brief

## 1. Research question

**Sharpened RQ** (one sentence, falsifiable):
针对特定区域的 PWV 辅助雷达端到端短临预测研究。

**Falsification condition** (what would you observe if FALSE):
CSI 下降明显；期望观察到引入 PWV 后 CSI 有提升，才能认为该方向有前途并继续。

**Smallest answerable version** (1-week prototype scope):
The researcher approved a 0--2 h pilot using 2,048 training windows and 512
validation windows. Compare a matched radar baseline with real PWV, null PWV,
and temporally reversed PWV using one contrastive-trigger checkpoint; do not
start full-data or multi-seed training unless this pilot passes its CSI gate.

## 2. Expected mechanism

**Causal chain**:
_TODO: articulate the expected moisture-to-birth/growth mechanism in the researcher's own words._

**Most uncertain step**:
_TODO_

**First step you'd bet breaks**:
_TODO_

## 3. Identifiability check

**Discriminating condition**:
The researcher approved checkpoint-only controls that isolate static moisture
level and spatial co-location. Real PWV must be compared with null PWV,
observed-input-only temporal reversal, an observed-period mean field repeated
over time (`level_only`), and a fixed half-domain spatial displacement.

**Confounders to rule out**:
- PWV interpolation and station-geometry shortcuts.
- Same-storm leakage across split boundaries.
- Model-dependent advection residual labels.

**Missing-data plan**:
_TODO: minimum acceptable GNSS coverage and missing-frame policy require researcher confirmation._

## 4. Validation plan

**Success metric**:
_TODO: pre-commit the primary metric and minimum effect size before viewing test results._

**Baseline being beaten**:
Matched `RAIN_2025_S` radar-only NowcastNet for final-field skill, plus an
architecture-matched zero-PWV Birth/Growth head for birth/growth PR-AUC.

**Negative control**:
Committed development controls are null PWV, observed-input-only temporal
reversal, `level_only`, and fixed spatial displacement. PWV climatology and a
station-distance mask remain candidates for the later full protocol.

**Implemented engineering gate (not a substitute for the researcher-authored
success-metric answer above)**:
The one-seed development pilot requires an absolute CSI improvement of at least
0.003 at both 10 and 20 mm/h versus null/radar identity and the train-time
spatial control, FAR degradation no larger than 0.005, and relative MAE
degradation no larger than 0.5%. Passing promotes only to three-seed,
day-cluster-bootstrap replication. The 30 mm/h threshold is diagnostic because
the Stage-0 audit found only eight positive training events.

## 5. Risk register

| # | Risk | Early-warning signal | Mitigation |
|---|---|---|---|
| 1 | PWV has no event-held-out incremental signal | Birth/growth PR-AUC is indistinguishable from shuffled PWV | Stop architecture expansion and report the negative pilot |
| 2 | Labels depend on the advection estimator | Conclusions reverse under alternate thresholds or flow baselines | Run threshold and advection sensitivity analysis |
| 3 | Too few independent events or GNSS stations | Wide event-bootstrap intervals or split-specific reversals | Add seasons/regions before manuscript claims |
| 4 | Split leakage across adjacent days | One storm spans train/val or val/test boundary | Manually move same-storm days to one split before locking manifest |
| 5 | PWV interpolation shortcut | Station-distance mask performs like PWV | Add geometry control and station-dropout tests |

## Notes

The retired executable protocol is
`code/protocols/pwv_contrastive_trigger_pilot.json`. The active successor is
`code/protocols/pwv_signed_calibrator_pilot.json`. Stage 0 found zero missing
PWV pairs, zero rejected non-contiguous windows, and a strong train/validation
class-distribution shift. It also found positive level/heavy-rain association
but negative short-slope/heavy-rain association. This brief remains draft
because the researcher-authored causal mechanism and final minimum-effect
answer have not yet been supplied. The implemented gate above is a reversible
engineering precommit for the pilot, not a fabricated researcher response.
