# Architecture and research decisions

## 2026-07-22 — Freeze the first publication-oriented comparison

**Decision.** Compare a `RAIN_2025_S` radar-only NowcastNet, a matched
zero-PWV Birth/Growth head, and a real-PWV `PWVBirthGrowthNowcastNet` under
`pwv_birth_growth_v1`. Use one manually
reviewed chronological day-block manifest, three matched seeds, complete test
sets, and identical evaluation thresholds.

**Identification constraint.** Initialize each Birth/Growth run from the
same-seed radar-only checkpoint and freeze the radar evolution and generative
backbone. Train only the PWV/fusion branch so the estimated increment can be
attributed to PWV rather than a separately learned motion field.

**Deferred.** V4 attention, FACL, object consistency, and further architecture
expansion are out of scope until this comparison establishes a PWV signal.

**Kill condition.** Stop expanding the model if event-held-out birth/growth
metrics fail to improve or later fail shuffled/climatology controls.

## 2026-07-22 — Lock meteorological boundaries and balance rare-event losses

**Split decision.** Use train through 2025-07-22, validation from 2025-07-23
through 2025-08-14, and test from 2025-08-15. This keeps the July 23--30 and
August 12--14 rain processes out of split boundaries.

**Sampling decision.** Full experiments remain exhaustive. Limited smoke runs
must sample uniformly across each chronological split rather than taking the
first windows, and must not be interpreted as scientific estimates.

**Loss decision.** Birth and growth pixels occur at roughly 1e-4 to 1e-3 rates
relative to the frozen evolution field. Normalize focal losses separately over
positive and negative pixels, and normalize source regression separately over
active and inactive pixels. Apply the same losses to real- and zero-PWV heads.

**Gate before PWV interpretation.** Do not interpret PWV deltas until the
matched radar-only checkpoint has non-zero validation/test skill at 10 mm/h.

## 2026-07-22 — Radar pilot passes the PWV-entry gate

**Evidence.** On 512 uniformly spaced test windows, the 10-epoch/2048-window
radar pilot achieved CSI 0.239 and 0.201 at 10 and 20 mm/h, versus persistence
0.169 and 0.123. Overall MAE/RMSE were 0.163/1.096 versus persistence
0.209/1.360. The best validation weighted-L1 occurred at epoch 7 (0.745).

**Decision.** Use this checkpoint only as the frozen backbone for a matched
2048-window PWV pilot. Do not promote it as the final full-data baseline.

**Core-control correction.** Disable PWV shuffle loss for both zero-PWV and
real-PWV in the primary comparison; otherwise PWV input would not be the only
experimental difference. Shuffle remains a later robustness control.

## 2026-07-22 - PWV Birth/Growth pilot is a no-go at aggregate scale

**Evidence.** Across 512 matched test windows, radar-only achieved MAE/RMSE
0.163/1.096 and CSI10/CSI20 0.239/0.201. The zero-PWV head degraded these to
0.244/1.244 and 0.221/0.183. Real PWV degraded them further to 0.288/1.622 and
0.204/0.118. Relative to zero PWV, real PWV reduced Birth PR-AUC from 0.01639
to 0.01464 and only increased Growth PR-AUC from 0.05106 to 0.05168, while
Birth/Growth false-alarm ratios remained 0.985/0.956.

**Decision.** Do not scale the current dense, non-negative additive source
head to the full dataset or multiple seeds. First run a checkpoint-only
horizon diagnostic, because the hypothesis specifically targets the second
and third forecast hours. A benefit confined to 2--3 h would justify a narrow
redesign; otherwise retire this formulation.

**Design implication.** Zero PWV is not an identity operation in the current
architecture: the source head also receives radar-derived features and can
produce a positive correction without PWV evidence. Any successor must make
the no-evidence path exactly or approximately identity-preserving and evaluate
PWV discrimination inside a physically motivated candidate region rather than
over every pixel.

## 2026-07-22 - Horizon diagnostic retains the PWV hypothesis

**Evidence.** In the 2--3 h bin, real PWV improved CSI10 from 0.05976 for the
matched zero-PWV head to 0.07023, and CSI20 from 0.01100 to 0.02149. It also
exceeded radar-only (0.01999 and 0.00010). The same model was harmful at 0--1 h
and did not consistently beat zero PWV at 1--2 h, so aggregate CSI remained
worse than radar-only.

**Decision.** Retain the scientific hypothesis that PWV contains useful
long-lead convective information, but retire the current all-lead dense source
injection as a publication candidate. The next model must be horizon-selective
and identity-preserving, with its primary endpoint declared as 2--3 h CSI and
with aggregate/early-lead degradation retained as safety metrics.

**Next gate.** Before full-data or multi-seed training, implement a small
matched pilot whose PWV residual is exactly zero under zero PWV and is disabled
or strongly constrained during 0--1 h. Require improvement over both radar and
zero-PWV controls at 2--3 h for CSI10 and CSI20 without material early-lead
degradation.

## 2026-07-22 - Correct primary scope to 0--2 h

**Scope correction.** The project target is 0--2 h precipitation nowcasting;
2--3 h performance is secondary and cannot justify the main method. Under the
primary scope, real PWV is strongly harmful at 0--1 h. At 1--2 h it improves
CSI10 over radar but not over zero PWV, and it is worse than both controls for
CSI20. Therefore the current experiment does not establish a PWV-attributable
CSI improvement for the project target.

**Revised decision.** Do not pursue a 2--3 h-selective architecture as the main
paper direction. Recompute exact pooled 0--2 h CSI/POD/FAR/Bias from existing
counts, treat the current PWV source formulation as no-go for 0--2 h, and
reconsider whether PWV should remain a primary modality or only a secondary
ablation in the next research design.

## 2026-07-22 - Adopt mechanism-named contrastive-trigger iteration

**Naming decision.** Stop using sequential version numbers for active research
iterations. Name each model by its testable mechanism and bind each run to a
machine-readable protocol. Historical version labels remain only as archive
metadata.

**Architecture decision.** Archive the dense additive Birth/Growth source as a
negative result. The active pilot is `contrastive_trigger`: radar proposes a
candidate/trigger region, while PWV contributes only positive evidence above
the same network's null-PWV response. Their AND-like product gates a residual,
so null PWV has an exact zero contribution and leaves the radar forecast
unchanged.

**Pilot gate.** In the pre-declared 0--2 h endpoint, real PWV must improve both
CSI10 and CSI20 over matched radar, null PWV, and temporally reversed PWV,
without an unacceptable false-alarm or frequency-bias increase. Only a passing
pilot may advance to full data and multiple seeds.
