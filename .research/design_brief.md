---
project: "NowcastNet with PWV Coupling"
last_updated: "2026-07-22"
stage: design
status: draft
source: "topic_dossier.gaps.yml#G1"
gap_verdict: "conditional-go — Proceed only if event-independent pilots show PWV adds robust information"
placeholder_segments: []
---

# Design brief

## 1. Research question

**Sharpened RQ** (one sentence, falsifiable):
Use gridded GNSS-PWV to predict only the positive birth/growth residual relative to a radar-advection baseline.

**Falsification condition** (what would you observe if FALSE):
_TODO: researcher confirmation required after the first frozen-protocol run._

**Smallest answerable version** (1-week prototype scope):
_TODO: researcher confirmation required; the implemented protocol compares a matched RAIN radar-only model with a frozen-backbone PWV birth/growth branch._

## 2. Expected mechanism

**Causal chain**:
_TODO: articulate the expected moisture-to-birth/growth mechanism in the researcher's own words._

**Most uncertain step**:
_TODO_

**First step you'd bet breaks**:
_TODO_

## 3. Identifiability check

**Discriminating condition**:
_TODO: confirm which shuffled/climatology controls will be treated as decisive._

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
_TODO: temporal shuffle, spatial shuffle, PWV climatology, and station-distance mask are proposed but not yet selected by the researcher._

## 5. Risk register

| # | Risk | Early-warning signal | Mitigation |
|---|---|---|---|
| 1 | PWV has no event-held-out incremental signal | Birth/growth PR-AUC is indistinguishable from shuffled PWV | Stop architecture expansion and report the negative pilot |
| 2 | Labels depend on the advection estimator | Conclusions reverse under alternate thresholds or flow baselines | Run threshold and advection sensitivity analysis |
| 3 | Too few independent events or GNSS stations | Wide event-bootstrap intervals or split-specific reversals | Add seasons/regions before manuscript claims |
| 4 | Split leakage across adjacent days | One storm spans train/val or val/test boundary | Manually move same-storm days to one split before locking manifest |
| 5 | PWV interpolation shortcut | Station-distance mask performs like PWV | Add geometry control and station-dropout tests |

## Notes

The executable protocol is `code/protocols/pwv_birth_growth_v1.json`; this brief remains draft because the Socratic mechanism, effect-size threshold, and negative-control commitment have not yet been supplied by the researcher.
