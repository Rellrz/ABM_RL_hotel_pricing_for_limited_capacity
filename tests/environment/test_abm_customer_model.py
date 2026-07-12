from __future__ import annotations

import numpy as np
import pandas as pd

from src.environment.abm_customer_model import HotelABMModel
from src.utils.preprocess_data import DemandCalibration


class RecordingPoissonRng:
    def __init__(self) -> None:
        self.lams: list[float] = []

    def poisson(self, lam: float) -> int:
        self.lams.append(float(lam))
        return int(round(float(lam)))


def make_model() -> HotelABMModel:
    model = HotelABMModel(historical_data=pd.DataFrame())
    model.calibration = DemandCalibration(
        weekday_arrival_mean=80.0,
        weekend_arrival_mean=40.0,
        ideal_offset_probs=np.asarray([0.5, 0.3, 0.2], dtype=float),
        weekday_ref_price=100.0,
        weekend_ref_price=120.0,
        offset_wtp_means=np.asarray([150.0, 140.0, 130.0], dtype=float),
        offset_wtp_stds=np.asarray([10.0, 10.0, 10.0], dtype=float),
    )
    return model


def test_arrivals_by_ideal_offset_use_stay_window_day_means() -> None:
    model = make_model()
    rng = RecordingPoissonRng()
    model.rng = rng

    arrivals = model._sample_arrivals_by_ideal_offset(current_day=4)

    assert rng.lams == [40.0, 12.0, 8.0]
    assert arrivals.tolist() == [40, 12, 8]


def test_arrivals_by_ideal_offset_do_not_spread_current_day_mean_to_future_offsets() -> None:
    model = make_model()
    rng = RecordingPoissonRng()
    model.rng = rng

    arrivals = model._sample_arrivals_by_ideal_offset(current_day=0)

    assert rng.lams == [40.0, 24.0, 16.0]
    assert arrivals.tolist() == [40, 24, 16]
