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

## 2026-07-23 - Contrastive-trigger pilot is weak-positive, replication required

**Evidence.** On 512 matched validation windows (seed 2026), real PWV exceeded
radar at CSI10 by 0.000762 (0.23% relative) and CSI20 by 0.001573 (0.77%
relative). It also exceeded null PWV and temporally reversed PWV at both
thresholds. Null PWV reproduced radar within numerical tolerance, confirming
the identity constraint.

**Caveat.** The gain was accompanied by higher POD and FAR: relative to radar,
POD increased by 0.00598/0.00386 and FAR by 0.00651/0.00204 at 10/20 mm/h.
MAE and RMSE worsened by 0.000827 and 0.000785. Temporal reversal retained most
of the CSI gain; correctly ordered PWV contributed only about 30% of the CSI10
increment and 25% of the CSI20 increment beyond radar. The result may therefore
be a small intensity/calibration shift rather than robust temporal PWV skill.

**Decision.** Classify the result as mechanism-positive but not full-data
ready. Advance to a matched three-seed replication gate using the same
2048/512 budget. Require consistent positive CSI deltas versus all controls,
report seed uncertainty, and inspect matched-FAR or matched-frequency-bias
skill before any full-data run.

## 2026-07-23 - Three-seed replication rejects full-data promotion

**Evidence.** Across seeds 2026--2028, mean real-minus-radar CSI deltas were
only +0.000300 at 10 mm/h and +0.000710 at 20 mm/h, with standard deviations
0.000710 and 0.000770. CSI10 reversed sign for seed 2028. Real PWV lost to
temporally reversed PWV at CSI10 for seed 2027 and at CSI20 for seeds 2027 and
2028; its mean advantage over reversal was only +0.000117/+0.000025. MAE was
worse in all three seeds (mean +0.000955), and RMSE was also worse on average.

**Training diagnosis.** For both newly supplied head-training curves, epoch 1
was the minimum validation weighted-L1. From epoch 1 to epoch 10, validation
support mean expanded roughly eightfold and validation false-alarm loss grew
about sevenfold while validation error worsened. The positive-only residual
therefore learns to activate increasingly broad support rather than isolate
rare, PWV-attributable initiation.

**Decision.** The `contrastive_trigger` formulation fails its replication gate
and must not be scaled to full data. Preserve the exact null identity and
paired-control machinery, but retire the positive-only additive amount path.
Before another training run, use checkpoint-only level-only and spatial-control
diagnostics to determine whether the small signal is static moisture state,
temporal evolution, or a geographical shortcut. A successor must allow signed
suppression as well as enhancement and must use a direction-specific control.

## 2026-07-23 - Correct temporal-control scope before interpretation

**Audit finding.** The original temporal-reverse implementation flipped all 29
paired PWV frames before the model selected its first nine inputs. It therefore
placed forecast-period PWV into the observed input slots. Radar, real-PWV, and
null-PWV results are unaffected, but every previously reported temporal-reverse
comparison is invalid and must be replaced.

**Correction.** All temporal and spatial controls now modify only the nine
observed PWV frames and leave the unused suffix untouched. Reports reject stale
temporal-reverse metrics unless they declare `observed_input_only`. Re-run the
corrected reverse control together with observed-mean `level_only` and fixed
half-domain spatial displacement for all three existing checkpoints; no model
training is required.

## 2026-07-23 - Diagnostics identify static local moisture as the dominant path

**Temporal result.** Real PWV was almost indistinguishable from `level_only`:
mean real-minus-level CSI deltas were +0.000012 at 10 mm/h and +0.000045 at
20 mm/h. Correct temporal order exceeded observed-input-only reversal by only
+0.000174/+0.000188, with uncertainty as large as or larger than the means.
Thus the final-field effect is not supported as a temporal-evolution benefit.

**Spatial result.** Spatial displacement reduced Birth PR-AUC from 0.00915 to
0.00547 and Growth PR-AUC from 0.02943 to 0.01434, showing that the head uses
local PWV/radar co-location. Real-minus-displaced CSI20 was positive in all
three seeds but remained tiny (+0.000430 mean, 0.000420 standard deviation),
while real PWV had worse MAE than the displaced control by 0.000542.

**Mechanistic verdict.** The current head primarily implements a static,
spatially aligned moisture-conditioned upward calibration. It does not establish
that PWV temporal evolution improves 0--2 h precipitation nowcasting, and its
small CSI trade is offset by false alarms and global error. Archive
`contrastive_trigger` as a completed no-go mechanism.

**Successor constraint.** If the PWV direction continues, the next pilot should
be a bounded signed moisture calibrator: exact null identity, both suppression
and enhancement, explicit 10/20 mm/h occurrence supervision, and a static-PWV
climatology/geography control. Treat temporal tendency as a separately gated
ablation rather than mixing it into the main path.

## 2026-07-23 - Literature review supports PWV feasibility but narrows novelty

**Positive feasibility evidence.** Direct prior art now includes Liu et al.
(IEEE TGRS 2025, DOI 10.1109/TGRS.2025.3554745), which reports a 26% relative
CSI gain at 30 mm/h from a Hong Kong radar/GNSS-PWV model; Lu et al. (IEEE TGRS
2025, DOI 10.1109/TGRS.2025.3587883), which combines radar QPE, satellite SWD,
and GNSS ZTD for 0--120 min nowcasting; and Sun et al. (Remote Sensing 2026,
DOI 10.3390/rs18121929), which reports +1 h torrential-rain CSI 0.409 versus
0.345 for its single-source model in Beijing--Tianjin--Hebei. A 2026 ZWD-Aurora
preprint also reports larger benefits in the heavy tail. These results make it
reasonable to continue the PWV direction.

**Novelty consequence.** A generic claim that adding PWV to radar improves CSI
is already occupied. Several reported gains bundle the data source with
attention, DEM/time inputs, satellite inputs, or a different loss, and report
point estimates or relative percentages without the controls needed to isolate
PWV. The defensible gap is therefore mechanism and identifiability: demonstrate
when spatially aligned PWV contains incremental event-held-out information over
radar, climatology, station geometry, and matched false-alarm calibration.

**Mechanism decision.** Make static/local PWV state (preferably a climatological
anomaly plus an explicitly retained absolute level) the primary condition.
Treat PWV temporal tendency as a secondary incremental ablation, because the
current project found real PWV nearly indistinguishable from `level_only`.
Require a radar-derived dynamic/uncertainty trigger. Do not return to a dense
non-negative source term.

**Next experiment class.** Before training, audit independent event counts,
threshold support, station geometry, and a leakage-safe climatology. Then run a
frozen-radar, small signed conditional calibration head with separate radar and
PWV encoders, exact null identity, explicit 10/20 mm/h occurrence losses, and
bounded support. Compare architecture-matched radar/null, real static/anomaly,
event-wise shuffled or displaced PWV, and real-plus-tendency variants. Set the
minimum meaningful effect after the support audit but before viewing model
results. Do not start the full-data experiment until the small pilot and a
three-seed/event-bootstrap replication pass.

## 2026-07-23 - Stage 0 locks thresholds and the signed-calibrator pilot

**Support audit.** The reviewed split contains 40/25/8 positive train events at
10/20/30 mm/h over 0--2 h, 18/16/14 validation events, and 13/9/7 test events.
All 29,511 radar frames have paired PWV frames and no candidate window failed
the six-minute continuity check. Validation and test are much heavier-rain
regimes than training, so window-level results must not be treated as
independent evidence.

**Feature verdict.** Deterministically sampled diagnostics show positive
association between absolute PWV level and future heavy-rain support but
negative association for the observed first-to-last PWV slope in both training
and validation. This independently supports static level/anomaly as the primary
condition and temporal tendency as a secondary ablation.

**Locked pilot.** Use 10 and 20 mm/h as primary thresholds; keep 30 mm/h
diagnostic. The successor is a frozen-radar bounded signed calibrator with
train-only spatial PWV climatology, exact null identity, a radar gate, and a
fixed candidate-support mask. Compare real static PWV against null and a
train-time spatial displacement control. A one-seed pass requires at least
+0.003 absolute CSI at both primary thresholds, FAR degradation <=0.005, and
relative MAE degradation <=0.5%; it promotes only to three-seed paired
day-cluster-bootstrap replication.
## 2026-07-23 — Replace recursive PWV source with latent state fusion

**Decision:** Stop tuning the signed recursive-source head. Do not retain
`Zero-PWV` as a separately named scientific control.

**Evidence:** On 512 validation windows, signed real PWV reduced CSI@10 by
0.0314, increased FAR@10 by 0.1457, and increased MAE by 71.2% relative to the
radar identity forecast. CSI@20 improved by 0.0067 but its day-cluster
bootstrap interval crossed zero. Real PWV did not clearly beat the spatial
control.

**Successor:** Encode observed PWV as a state, fuse it with radar at the
generative latent using cross-attention, and train end-to-end at a reduced
radar learning rate. Compare aligned PWV with separately trained radar-only
and train-time displaced-PWV models.

**Gate:** Do not run multiple seeds or locked-test/full-data experiments unless
aligned PWV improves CSI@10 and CSI@20 by at least 0.003 against both controls
without FAR increasing more than 0.005 or MAE increasing more than 0.5%.

## 2026-07-23 — Latent-state pilot verdict

**Decision:** Do not promote the current latent-state model to multiple seeds
or full-data training. Preserve its checkpoints for intervention diagnostics.

**Evidence:** All variants evaluated the same 512 windows with identical sample
hashes. Against radar-only, aligned PWV changed CSI@10 by +0.00342 but CSI@20
by -0.03199. The latter day-cluster bootstrap interval was entirely negative
[-0.04580, -0.02410]. Against the independently trained spatial-displacement
control, aligned PWV changed CSI by only +0.000004 at 10 mm/h and -0.000120 at
20 mm/h; both bootstrap intervals crossed zero. Aligned and displaced MAE also
differed by less than 0.000009.

**Interpretation:** The 7.25% MAE reduction versus the continued radar baseline
cannot be attributed to geographically aligned PWV, because the displaced-PWV
model obtained the same reduction. The fusion gate mean was nonzero, but the
effective PWV-dependent perturbation was negligible. Checkpoint selection by
weighted-L1 at epoch 1 also conflicts with the primary CSI@10/20 objective.

**Next diagnostic:** Evaluate the same aligned checkpoint twice, using aligned
and deterministically displaced observed PWV. This within-checkpoint
intervention removes optimization-trajectory confounding. Do not retrain until
prediction sensitivity and fusion-residual magnitude have been measured.

**Same-checkpoint result:** The intervention confirmed functional PWV
insensitivity. On the same 512 samples, aligned-minus-shifted CSI was
-0.000078 at 10 mm/h and +0.000026 at 20 mm/h. The 10,000-repeat day-cluster
bootstrap intervals were [-0.000167, +0.000006] and
[-0.000068, +0.000164], respectively. Mean paired event MAE changed by only
0.000036 and the largest absolute event-level MAE change was 0.000835.

**Closure:** Close this latent-fusion candidate. Its nonzero gate value is not
evidence of PWV use. Before another nowcasting architecture is built, run a
conditional-information probe asking whether observed PWV adds predictive
information for future 10/20 mm/h occurrence after radar history is already
known.

## 2026-07-23 — Conditional-information probe protocol

**Decision:** Freeze the matched radar evolution path and generative encoder,
then predict per-lead 10/20 mm/h occurrence on its 1/8-resolution latent grid.
This is an information diagnostic, not the final forecasting architecture.

**Fairness:** Radar-only and radar+PWV probes instantiate the same architecture
and parameter count (13,378 parameters with the server's `ngf=32`; 6,210 in
the light-channel local smoke). Radar-only uses a learned constant auxiliary slot.
The PWV checkpoint is evaluated unchanged with aligned, half-domain-shifted,
and deterministic cross-event PWV. Fit and probability-threshold calibration
use disjoint training days; validation is untouched.

**Promotion:** At least three of four horizon-threshold tasks must improve CSI
by at least 0.003 and average precision by more than zero against radar-only
and both same-checkpoint controls. No task may lose more than 0.003 CSI.

## 2026-07-23 - Conditional-information probe verdict

**Decision:** Do not promote the present gridded-PWV representation to a
restricted joint-adaptation, multi-seed, or full-data experiment. The
pre-registered gate failed on all four horizon-threshold tasks.

**Integrity checks:** The server run used 2,048 training and 512 untouched
validation windows, with 1,628 fit and 420 calibration samples from disjoint
training days. Radar-only and radar+PWV probes each had 13,378 trainable
parameters over the same 35,035,965 frozen radar parameters. Both optimizations
converged; final fit losses were 0.0670 and 0.0626, respectively.

**Primary evidence:** Aligned PWV minus radar-only CSI was -0.02139,
-0.02563, -0.00002, and -0.03469 for 0--1 h at 10/20 mm/h and 1--2 h at
10/20 mm/h. Average-precision deltas were also negative in all four tasks.
The day-cluster bootstrap intervals excluded zero on the harmful side for
0--1 h at 10 mm/h and 1--2 h at 20 mm/h.

**Control interpretation:** At 1--2 h, aligned PWV beat spatially shifted PWV
by +0.02279 CSI at 10 mm/h (95% day-cluster interval
[+0.00954, +0.03544]) and +0.01078 at 20 mm/h, but it still did not beat
radar-only. Cross-event controls were not consistently worse. This is evidence
of some location-sensitive PWV response, not evidence of incremental forecast
skill after radar history is known.

**Next gate:** Stop architecture search temporarily. Audit geospatial
co-registration, PWV physical scaling/inversion, temporal cadence, and
event-regime heterogeneity. Only reopen model development if a pre-specified
radar-observable initiation/rapid-growth subgroup or a corrected physical PWV
representation shows reproducible conditional information beyond radar-only.

**Temporal-audit refinement:** Local data inspection confirms complete
radar/PWV filename pairing and matching 70x66 raster geometry. The six-minute
PWV frames are extremely autocorrelated and are largely reconstructable from
30-minute endpoints. Therefore, failure of the nine-frame PWV probe does not
fully test the physically better hypothesis of multi-hour moisture
preconditioning. Permit one further matched-capacity information diagnostic:
retain the existing nine-frame radar history, but encode native-cadence PWV anchors over
approximately three hours as level, train-only climatological anomaly,
multi-hour tendency, and spatial gradient. Evaluate it first on all windows and
on a pre-specified radar-observable initiation/rapid-growth stratum. This is a
data-representation test, not permission for unrestricted architecture search.

**Locked successor implementation:** Name the hypothesis rather than assigning
a version number: `pwv_causal_preconditioning_probe`. Use seven 30-minute
anchors over three hours, ending no later than the radar issue time. The
primary mechanism stratum is defined exclusively from observed radar:
1 mm/h <= last tile maximum < task threshold and last tile maximum no lower
than the first-three-frame mean. The stratum has 885--2,955 validation positive
tile-leads per task in the locked 512-window development sample. Promotion
requires at least three of four primary-stratum tasks to beat radar-only,
short-interpolated PWV, spatial shift, and cross-event PWV by >=0.003 CSI with
positive AP delta, while losing no more than 0.003 all-window CSI to radar-only.
