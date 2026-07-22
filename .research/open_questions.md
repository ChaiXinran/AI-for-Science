# Open questions

1. What exact dates/events are assigned to train, validation, and test, and are splits event-disjoint?
2. Why do the archived radar/V2 runs target `RADAR_2025_S` while V3/V4 target `RAIN_2025_S`? What are the units and calibration of each?
3. Do all 24-sample recomputed runs contain identical sample IDs? Persistence differences show that at least the V2/radar family is not aligned with V3/V4.
4. How many independent storm events, seasons, GNSS stations, and valid station-hours exist after quality control?
5. Are PWV fields generated using information available strictly before forecast issue time?
6. Does gridded PWV add skill beyond radar history, seasonal climatology, topography, station geometry, and shuffled PWV?
7. Are convective birth/growth labels stable across advection algorithms, thresholds, neighborhood radii, and object-persistence filters?
8. Which code commit produced each remote run? The archive records configs but not a code commit hash per experiment.
9. What checkpoint-selection metric was used for `best.ckpt`, and was it chosen without test-set access?
10. Is the publication target a specific CCF-A venue, and does its scope accept meteorological application papers without a broadly reusable ML method?
