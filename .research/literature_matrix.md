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
