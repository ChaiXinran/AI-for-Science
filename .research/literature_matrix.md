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
