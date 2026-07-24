# PWV integration literature matrix

This matrix focuses on how an auxiliary moisture modality should change a dense
0-2 h precipitation nowcast. The previous copy of this file was replaced
because its text encoding was corrupted and unreadable.

## Full-text checked in this pass

| Citation | Forecast setting and modalities | Integration / objective | Main evidence | Main limitation | Implication for this project | Use as |
|---|---|---|---|---|---|---|
| Yu et al., 2025, *Integrating Multi-Source Data for Long Sequence Precipitation Forecasting* (AAAI-25) | 3 h radar forecast from radar plus satellite sequences on SEVIR | Separate modality branches; latent spatial-temporal multimodal attention; frame-index-conditioned Temp-AdaLN; deterministic predictor followed by a flow-based distribution adaptor | Removing multimodality reduced mCSI from 0.2562 to 0.2377; removing Temp-AdaLN reduced it to 0.2488. The paper attributes the gain to latent alignment and lead-dependent adaptation rather than channel concatenation | Uses satellite rather than GNSS PWV, targets 3 h, and combines several components, so it does not isolate moisture causally | Keep radar and PWV encoders separate; fuse in latent space; make the PWV effect lead-dependent; do not inject an unconditional dense source at every lead | Architecture precedent and ablation design |
| Yao, Shan, and Zhao, 2017, *Establishing a method of short-term rainfall forecasting based on GNSS-derived PWV and its application* (Scientific Reports, DOI 10.1038/s41598-017-12593-z) | Station-level short-term rainfall occurrence from hourly GNSS PWV | Three factors: location/month-conditioned absolute PWV, PWV variation, and PWV rate of change; rate of change is primary, other factors reduce misses | PWV commonly rises before rain, but high PWV alone also occurs without rain. The paper reports that the combined factors improve correct rate over a single rate-of-change factor, while false alarms remain substantial | Station-level occurrence rather than dense radar fields; hourly resolution; empirical thresholds; no independent dynamic trigger such as CAPE or radar growth | Preserve absolute/climatological PWV, variation, and rate as separate signals. Treat PWV as a necessary moisture condition and require a radar-derived trigger. A PWV-only or OR gate is physically wrong and invites false alarms | Physical mechanism and feature design |
| Miralles et al., 2026, *Pointwise is Pointless? A Multimodal Ablation Study for Precipitation Nowcasting with Graph Neural Networks* (arXiv:2606.18436v2) | 0-2 h, 5 min Nordic radar-grid forecasts using radar, NWP, stations, satellite, noise, and CRPS ensembles | Multimodal GNN ablations; deterministic and ensemble/CRPS objectives; radar-grid, onset, station, oracle, displacement, and amplitude diagnostics | Different modalities improve different forecast properties. Direct auxiliary inputs can improve local/onset diagnostics without consistently improving the dense radar field; deterministic satellite input can cause premature rain activation. CRPS configurations give the most consistent radar-grid improvements | 2026 preprint; modalities and graph geometry differ from gridded GNSS PWV; does not provide a ready-made PWV architecture | Do not regress a dense rain-rate correction directly from moisture evidence. Use PWV as a constraint on candidate objects or on a probabilistic residual distribution; evaluate onset, FAR, bias, and spatial coherence separately from global pixel loss | Objective/verification precedent and warning against direct fusion |

## Cross-paper synthesis

1. Auxiliary moisture data are useful only after their spatial support and time
   semantics are aligned with radar. Separate encoders plus latent fusion are
   better motivated than raw channel concatenation.
2. Absolute PWV, climatological anomaly, accumulated variation, and rate of
   change carry different information. Per-sample standardization alone removes
   part of the physically meaningful moisture-sufficiency signal.
3. Moisture is necessary but not sufficient for convection. The model should
   implement an AND-like interaction between PWV sufficiency and a radar-derived
   dynamic trigger, not an OR-like activation.
4. Extra modalities can improve onset or local constraints while degrading the
   dense field through premature activation. CSI must be accompanied by
   horizon-wise POD, FAR, bias, and object/neighborhood diagnostics.
5. Lead-dependent conditioning and probabilistic/constraint-based objectives
   are the strongest reusable ideas. A dense non-negative source active at all
   leads is poorly supported by this literature.

## Folder routing for the next reading pass

| Priority | Local paper group | Question to resolve | Status |
|---|---|---|---|
| 1 | `Revealing the synergistic contribution of PWV and CAPE...` | Which thermodynamic trigger should complement moisture sufficiency? | Not full-text checked this pass |
| 2 | `Bayesian Deep Learning for Convective Initiation...` | How should rare initiation probability and uncertainty be calibrated? | Not full-text checked this pass |
| 3 | `MoCast`, `Fully Differentiable Lagrangian...` | Where should a constrained source/sink enter a motion-evolution model? | Not full-text checked this pass |
| 4 | `fourier-amplitude-and-correlation-loss`, `Hybrid physics-AI...` | Which spatial/intensity objectives improve CSI without excessive activation? | Not full-text checked this pass |

## Verified additions — 2026-07-23

The three rows marked **deep read** were checked against the complete local PDF.
The remaining rows were checked against the publisher/author abstract and the
methods/results text exposed by the official page. Relative percentage gains
are not treated as absolute CSI-point gains.

| Citation | Forecast setting and modalities | Integration / objective | Main evidence | Main limitation | Implication for this project | Use as |
|---|---|---|---|---|---|---|
| Liu et al., 2025, *Revealing the synergistic contribution of PWV and CAPE to extreme precipitation throughout China* (Advances in Space Research, DOI 10.1016/j.asr.2024.11.050) **deep read** | Daily PWV, CAPE and extreme precipitation at 219 collocated Chinese GNSS stations, 2011–2018 | Statistical regional/seasonal analysis of PWV–CAPE–extreme-rain relationships | High PWV and CAPE are complementary; their joint state is more informative than either variable alone, with strong geographic and seasonal dependence | Daily association, not 0–2 h causal evidence; CAPE is not currently in this project's dataset | PWV should condition a dynamic/instability trigger rather than act as a rain source by itself; season/climatology controls are mandatory | Physical mechanism, not architecture evidence |
| Fan et al., 2026, *Bayesian Deep Learning for Convective Initiation Nowcasting Uncertainty Estimation* (AI for Earth Systems, DOI 10.1175/AIES-D-25-0064.1) **deep read** | 0–1 h GOES-16 convective-initiation probability | Initial-weight ensembles, MC dropout and Bayesian-MOPED; probability calibration and uncertainty | Initial-weight ensemble plus MC dropout gave the best calibrated CI probabilities; clear-sky and anvil false alarms remained important failure modes | Satellite CI rather than rain-rate fields; no PWV | Treat PWV-sensitive initiation as rare probabilistic classification; verify Brier/BSS, reliability and event cases before allowing it to modify dense rainfall | Objective and uncertainty precedent |
| Trentini et al., 2026, *Integrating GNSS-Derived Zenith Wet Delay into a Weather Foundation Model Improves Precipitation Forecasting* (arXiv:2607.05658) **deep read** | Aurora fine-tuned for 6 h precipitation with gridded ZWD | ZWD is introduced as an atmospheric state variable during downstream fine-tuning | ETS gain increased with severity and reached 8.8% at the 99th percentile; paired checkpoint tests reported p <= 0.013; spectra and spatial organization also improved | 6 h foundation-model setting; ZWD grid is partly ERA5-dependent and smoothed; does not isolate level versus tendency | Strong evidence that column moisture can help the heavy tail as a spatial condition; weak evidence for PWV tendency as the operative mechanism | Positive feasibility evidence and evaluation precedent |
| Liu et al., 2025, *A Deep Learning-Based Precipitation Nowcasting Model Fusing GNSS-PWV and Radar Echo Observations* (IEEE TGRS, DOI 10.1109/TGRS.2025.3554745) | Regional Hong Kong radar plus GNSS-PWV nowcasting | Multi-source fusion plus time-dimension attention | Publisher/author abstract reports 26% relative CSI and 23% relative HSS improvement at 30 mm/h for the complete model | Full text was not available locally; abstract does not expose absolute scores, event split, uncertainty, or controls that isolate PWV from the attention change | Directly establishes viability but removes novelty from a generic “radar + PWV fusion” claim; our contribution must be stronger controls, mechanism and generalization | Closest prior art and novelty boundary |
| Lu et al., 2025, *RSG-GAN: A GAN-Based Precipitation Nowcasting Model Integrating Radar QPE, GOES-16 SWD, and GNSS ZTDs* (IEEE TGRS, DOI 10.1109/TGRS.2025.3587883) | 0–120 min over the US west coast; radar QPE, GOES-16 split-window difference and GNSS ZTD | Multisource GAN; radar-only and radar–satellite baselines | Across 189 precipitation periods, abstract reports MAE 0.34 mm/h and RMSE 0.61 mm/h, plus higher-threshold CSI/FSS gains and a SEVIR transfer experiment | GNSS contribution is bundled with satellite input and a GAN; optical-flow percentage reductions do not isolate GNSS value | A second direct competitor: null-PWV, shuffled/shifted-PWV and matched-capacity radar controls are necessary for a defensible causal claim | Closest multimodal competitor and control-design warning |
| Sun et al., 2026, *Synergistic Fusion of GNSS-PWV and Radar for Precipitation Nowcasting: An AI-Empowered Spatio-Temporal Attention Network* (Remote Sensing, DOI 10.3390/rs18121929) | Beijing–Tianjin–Hebei, May–August 2025; 60 min radar/PWV history; +1/+2/+3 h outputs | PWV, radar, DEM and time encoding in a Swin U-Net; CBAM; Huber + Dice + edge loss | At +1 h “torrential rain,” fused CSI 0.409 versus 0.345 for the single-source model; POD 0.676 versus 0.556 | One flood season, 0.1 degree grid, PWV resampled from 30 to 6 min, point estimates only; the reported fusion package includes PWV, DEM, time encoding and loss/architecture choices, so PWV is not cleanly isolated | Highly relevant regional evidence but also a warning: reproduce the gain under matched capacity and real/null/reverse/shift/anomaly controls; avoid claiming interpolation-created 6 min PWV dynamics | Positive evidence, competing paper and audit checklist |
| Rohm et al., 2020, *GNSS-Based Machine Learning Storm Nowcasting* (Remote Sensing, DOI 10.3390/rs12162536) | 0–2 h storm occurrence in Poland from IWV and 3-D wet refractivity | Random-forest binary classifier using surrounding/layered moisture state and evolution | Accuracy exceeded 87%, but precision was about 30%; low-level wet refractivity and upwind grids ranked highly | Lightning-defined storm voxels, two summer seasons, coarse classification; severe class imbalance and many false alarms | Moisture geometry and upstream structure matter, but PWV-family predictors alone overactivate; use radar trigger and precision/FAR gates | Mechanism and false-alarm precedent |
| Li et al., 2022, *An Improved Method for Rainfall Forecast Based on GNSS-PWV* (Remote Sensing, DOI 10.3390/rs14174280) | 12 h station rainfall occurrence at 66 US stations | Threshold voting over PWV value, increase and maximum hourly increase | Mean POD about 87%, but mean FAR about 53%; feature importance depends on climatic region | Station occurrence, long horizon and hand-tuned thresholds | Keep level and tendency separate; do not assume a universal PWV threshold; high POD without FAR control is not success | Feature/control precedent |
| Liu et al., 2023, *A novel rainfall forecast model using GNSS observations and CAPE in Singapore* (JASTP, DOI 10.1016/j.jastp.2023.106158) | Hourly station rainfall using PWV, CAPE and temperature | Support-vector regression | Reports strong station-level fit and reinforces PWV–CAPE complementarity | Four stations, regression setting, not dense 0–2 h radar nowcasting | If CAPE is unavailable, the radar-derived growth/instability feature must play the trigger role explicitly | Physical-feature precedent |
| Shin et al., 2024, *Improvements in deep learning-based precipitation nowcasting using major atmospheric factors with radar rain rate* (Computers & Geosciences, DOI 10.1016/j.cageo.2024.105529) | 1–6 h Korean radar rain rate plus total-column water vapour and 925 hPa divergence | Encoder–forecaster; categorical-score-aware loss | Extra atmospheric factors improved moderate/heavy ETS; the best configuration highlighted low-level divergence and could capture some new growth | Reanalysis variables and longer leads; best result does not establish water vapour alone as sufficient | Moisture plus a convergent/dynamic trigger is better supported than moisture-only source generation; threshold-aware objectives are justified | Architecture/objective and mechanism precedent |
| Ye et al., 2026, *Improving precipitation nowcasting via multiphysical parameter fusion in radar echo extrapolation* (Journal of Hydrology, DOI 10.1016/j.jhydrol.2026.134947) | Co-registered reflectivity plus four polarimetric radar variables | Separate intra-/inter-modal branches, attention fusion and gated spatiotemporal attention | Publisher abstract reports up to 24.2% relative CSI improvement for >=40 dBZ | Extra variables are perfectly co-registered radar observables, unlike sparse/smoothed PWV | Separate encoders and interaction-aware fusion are supported; the PWV resolution mismatch needs explicit treatment and controls | Fusion-architecture precedent |

## Decision synthesis after the verified additions

1. **Feasibility is positive, but the generic claim is occupied.** At least two
   2025 IEEE TGRS papers and one 2026 regional paper report gains from
   radar+GNSS moisture fusion. A publishable claim cannot be merely that PWV is
   added to radar and CSI rises.
2. **The best-supported primary role is a spatially local moisture state, not a
   dense positive source and not yet a PWV tendency claim.** This agrees with
   this project's level-only diagnostic. Tendency should remain a prespecified
   ablation until it beats the static/anomaly control.
3. **The missing scientific contribution is identifiability.** A matched radar
   baseline, real PWV, null PWV, climatology/level-only PWV, temporal reverse,
   spatial shift, and (where valid) event-wise PWV permutation should be run
   from the same initialization and training budget. Architecture changes must
   not be bundled with the data-source ablation.
4. **The next model should be a conditional classifier/calibrator around a
   radar forecast, not another unconstrained generator.** Use separate encoders;
   let PWV modulate signed threshold logits or a bounded residual only where a
   radar-derived growth/uncertainty trigger is active. Do not allow PWV alone to
   create widespread positive rain.
5. **Primary endpoints should be heavy-rain event skill and calibration.** Use
   horizon-wise CSI/POD/FAR/bias at 10 and 20 mm/h (30 mm/h only if enough
   positives), FSS/object diagnostics, reliability/Brier score for the trigger,
   multi-seed paired deltas, and event-bootstrap confidence intervals.

## Post-pilot correction: fusion location and control design

The signed-source pilot falsified the proposed recursive source-calibration
mechanism. Real PWV increased 10 mm/h FAR by 0.146, reduced CSI by 0.031, and
increased MAE by 71% relative to the identity radar forecast. This requires an
architecture-level correction, not hyperparameter tuning.

| Work | Where the auxiliary modality enters | How it is trained | Appropriate no-auxiliary comparison | Reusable lesson |
|---|---|---|---|---|
| Liu et al., TGRS 2025, radar + GNSS-PWV | Multi-source fusion with time-dimension attention | End-to-end precipitation forecasting | Radar-only model | Direct prior art supports learned latent fusion, but the available abstract does not justify a recursive PWV rain source |
| Sun et al., Remote Sensing 2026, STEA-Swin | PWV temporal representation and radar spatial representation interact through Swin/spatio-temporal attention; DEM/time encoding and an edge-aware composite loss are also included | End-to-end multimodal U-Net | Single-source radar model | Their reported gain is a bundled fusion package; reproduce the basic two-stream latent-fusion idea under stricter matched controls |
| Trentini et al., 2026, ZWD + Aurora | ZWD is added as a new surface state variable through the variable-specific linear encoder/decoder and is also an auxiliary prediction target | Two-step fine-tuning; matched With-ZWD and Without-ZWD models start from the same pretrained checkpoint and share schedule/data/budget | Separately trained Without-ZWD model, not a zeroed ZWD inference | Treat water vapour as a state representation and auxiliary task; compare independently trained matched models |
| Yu et al., AAAI 2025, radar + satellite | Separate encoder-decoder branches, latent cross-attention, and lead-dependent Temp-AdaLN; a later flow adaptor refines the distribution | Joint recurrent multimodal prediction, then distribution adaptation | Full model versus model trained without multimodality | Fuse encoded modalities and make their influence lead-dependent rather than adding a dense physical source |
| Yu et al., ACM MM 2025, PiMMNet | Multimodal inputs estimate a shared velocity field; deterministic advection is separated from stochastic residual generation | Jointly optimized motion estimator and residual diffusion model | Matched pipeline/input-condition ablations | Auxiliary observations can inform dynamics or residual uncertainty without being interpreted as a direct positive precipitation source |

### Corrected control decision

- Drop `Zero-PWV` as a separately named scientific experiment. Under the
  current exact-identity construction it is only a duplicate of the frozen
  radar forecast, while zeroing a modality at inference can also be an
  out-of-distribution intervention.
- Keep one radar-only baseline, trained under the same 0-2 h split and budget.
- Compare it with a separately trained radar+PWV model initialized from the
  same radar checkpoint.
- Use train-time event-wise PWV permutation or fixed spatial displacement as
  the negative control for whether real PWV content/alignment is informative.
- Do not inject a per-step PWV source into the recursive evolution. The next
  candidate should use two-stream latent fusion and, if a residual is retained,
  apply a single bounded correction to the final forecast rather than allowing
  twenty-step accumulation.

## Parameter audit of PWV/GNSS multimodal models

Parameter balance is generally not reported as a modality-by-modality quantity,
and the reviewed papers do not support requiring equal radar and PWV branch
sizes.

| Study | Reported total capacity | PWV/GNSS-specific capacity | What is verifiable |
|---|---:|---:|---|
| Trentini et al. 2026, ZWD into Aurora | ~1.3B (main); ~110M (small ablation) | ~24,592 parameters for one added input/output surface variable in the 1.3B configuration; ~12,304 in the 110M configuration | Derived from the public Aurora implementation: one variable-specific `LevelPatchEmbed` kernel plus one linear patch-reconstruction head. The shared backbone is fine-tuned and no bespoke ZWD branch is added. |
| Sun et al. 2026, STEA-Swin | Not reported in the article | Not separately identifiable | PWV, radar, DEM and time encodings enter a shared end-to-end Swin U-Net; the paper reports no per-modality parameter accounting. |
| Liu et al. 2025, radar + GNSS-PWV TGRS | Not reported in accessible abstract/metadata | Not reported | Multi-source fusion and temporal attention are described, but the accessible sources do not expose layer widths or a parameter table. |
| Lu et al. 2026, GRENet | Not reported in accessible main text | Not reported | A dual-stream GNSS projector and radar encoder are described; detailed layers are delegated to supporting information, without a parameter total in the accessible text. |
| Lu et al. 2025, RSG-GAN | Not reported in accessible abstract | Not reported | GNSS ZTD and satellite information are bundled, so a GNSS-only parameter count cannot be isolated from the published abstract. |

For this project, the current trainable counts are 34,932,274 radar-path
parameters and 615,616 PWV/fusion parameters (56.7:1). This imbalance can
confound comparisons between separately trained models, but it cannot explain
the same-checkpoint aligned-versus-shifted invariance: that intervention holds
all parameters fixed and directly shows that the learned predictor is
functionally insensitive to PWV location.

## Radar-pivot screening: interpretability and geographic generalization

This section evaluates the two directions articulated by the project lead on
2026-07-24. It does not reopen dynamic-PWV modelling on the current interpolated
PNG product.

| Citation | Problem and data | Relevant method/evidence | Limitation for this project | Decision relevance |
|---|---|---|---|---|
| Miralles et al., 2026, *Pointwise is Pointless?* (arXiv:2606.18436) | Nordic 0–2 h multimodal nowcasting from radar, NWP, stations and satellite | Source-by-source ablations show that each modality improves different targets; station/onset gains do not automatically become coherent radar-grid gains, and deterministic satellite input can activate rain too early | Preprint; not GNSS-PWV and not a cross-region study | Strong caution against treating a CSI change or an attention map as an explanation; motivates target-specific attribution and uncertainty |
| Song et al., 2025, *Prior Information Assisted Multi-Scale Network for Precipitation Nowcasting* (DOI 10.1016/j.cageo.2025.105851) | Radar nowcasting with terrain and cross-region pretrained knowledge | Cross-attention relates terrain elevation to radar features; a teacher model from another region guides the target-region student | Requires target-region training and bundles terrain, architecture and distillation; not a leave-one-region-out identification of invariant dynamics | Directly occupies generic “add terrain/transfer knowledge” but supports explicit static-region conditioning |
| Zhou et al., 2026, *Learning to Refine: Spectral-Decoupled Iterative Refinement* (OpenReview zB4xF9tfdm) | Radar nowcasting on CIKM, Shanghai and SEVIR | Reports a zero-shot transfer from SEVIR to Shanghai and attributes robustness to deterministic frequency-decoupled refinement | Very recent; one transfer direction and no terrain/PWV conditioning | Establishes a necessary cross-domain baseline and means “cross-region generalization” alone is not a sufficient novelty claim |
| Ravuri et al., 2021, *Skilful Precipitation Nowcasting Using Deep Generative Models of Radar* (DOI 10.1038/s41586-021-03854-z) | UK radar nowcasting with a separately trained and evaluated US dataset | Demonstrates that an architecture can remain competitive on a second radar domain; evaluates CSI, spectra, CRPS and expert utility | It retrains on the US data rather than testing geographic zero-shot transfer | Shows that multi-region evidence is an established expectation and that CSI alone is inadequate |
| Lu et al., 2026, *GRENet* (DOI 10.1029/2025GL120787) | GNSS water vapour and radar GAN nowcasting | Reports better CSI/FSS and more accurate heavy-rain extent/location than radar-only for a heavy-rain case | Case-study evidence and accessible page do not establish matched attribution or cross-region robustness | The broad GNSS-enhanced radar claim is occupied; novelty must come from identifiable modality roles and generalization |

### Screening synthesis

1. The two proposed directions are scientifically compatible but should not be
   implemented as one large architecture before data feasibility is known.
2. The publishable intersection is not “more modalities”. It is an explicit
   separation of fast radar dynamics, static geographic conditions and slowly
   varying environmental state, tested by counterfactual modality controls.
3. Cross-region generalization currently fails the data gate: the audited
   project contains one confirmed regional dataset. Random crops, dates or
   seasons are not substitutes for a held-out geographic domain.
4. The current interpolated PWV product may remain a slow environmental or
   climatological covariate, but it cannot support another claim about
   independently observed minute-scale PWV dynamics.
