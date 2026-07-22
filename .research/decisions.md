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
