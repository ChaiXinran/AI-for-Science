import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.object_attribution import analyze_sample, summarize_records


def _square(field, y, x, value, size=3):
    field[y:y + size, x:x + size] = value


def test_birth_and_oracle_attribution():
    last_input = np.zeros((24, 24), dtype="float32")
    pred = np.zeros((2, 24, 24), dtype="float32")
    target = np.zeros_like(pred)
    _square(target[0], 8, 8, 20.0)
    _square(target[1], 9, 8, 24.0)

    record = analyze_sample(
        last_input,
        pred,
        target,
        thresholds=(10.0,),
        change_fractions=(0.2,),
        frame_minutes=30.0,
        horizon_bins=((0.0, 1.0, "0-1h"),),
        min_area=4,
    )
    summary = summarize_records([record])
    values = summary["thresholds"]["10"]["0.2"]["0-1h"]
    assert values["regimes"]["birth"]["observed"] >= 1
    assert values["regimes"]["birth"]["missed"] >= 1
    assert values["oracles"]["birth_existence"]["csi_delta_vs_original"] > 0


def test_displacement_oracle_improves_shifted_object():
    last_input = np.zeros((24, 24), dtype="float32")
    pred = np.zeros((1, 24, 24), dtype="float32")
    target = np.zeros_like(pred)
    _square(last_input, 8, 4, 20.0)
    _square(pred[0], 8, 7, 20.0)
    _square(target[0], 8, 10, 20.0)

    record = analyze_sample(
        last_input,
        pred,
        target,
        thresholds=(10.0,),
        change_fractions=(0.2,),
        frame_minutes=60.0,
        horizon_bins=((0.0, 1.0, "0-1h"),),
        min_area=4,
        max_distance_pixels=8.0,
    )
    values = summarize_records([record])["thresholds"]["10"]["0.2"]["0-1h"]
    assert values["oracles"]["displacement"]["csi_delta_vs_original"] > 0


def test_intensity_oracle_recovers_underforecast():
    last_input = np.zeros((24, 24), dtype="float32")
    pred = np.zeros((1, 24, 24), dtype="float32")
    target = np.zeros_like(pred)
    _square(last_input, 8, 8, 20.0)
    _square(pred[0], 8, 8, 12.0)
    _square(target[0], 8, 8, 24.0)

    record = analyze_sample(
        last_input,
        pred,
        target,
        thresholds=(20.0,),
        change_fractions=(0.2,),
        frame_minutes=60.0,
        horizon_bins=((0.0, 1.0, "0-1h"),),
        min_area=4,
    )
    values = summarize_records([record])["thresholds"]["20"]["0.2"]["0-1h"]
    assert values["oracles"]["intensity"]["csi_delta_vs_original"] > 0
