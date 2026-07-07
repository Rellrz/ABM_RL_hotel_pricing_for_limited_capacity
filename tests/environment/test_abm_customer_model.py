from __future__ import annotations

import numpy as np
import pandas as pd

from configs.config import ABM_CONFIG
from src.environment.abm_customer_model import HotelABMModel


def _minimal_historical_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hotel": ["City Hotel"],
            "lead_time": [0],
            "adr": [100.0],
            "arrival_date_year": [2017],
            "arrival_date_month": ["January"],
            "arrival_date_day_of_month": [2],
        }
    )


def _make_single_customer_model(customer_type: str) -> HotelABMModel:
    model = HotelABMModel(historical_data=_minimal_historical_data(), random_seed=1)
    model._sample_arrivals = lambda current_day: 1
    model._sample_customer_type = lambda: customer_type
    model._sample_ideal_offset = lambda: 0
    model._sample_wtp = lambda stay_day: 100.0
    return model


def test_flex_customer_reselects_available_date_when_first_choice_is_full() -> None:
    original_noise = float(ABM_CONFIG.utility_noise_std)
    ABM_CONFIG.utility_noise_std = 0.0
    try:
        model = _make_single_customer_model("flex")
        result = model.simulate_day(
            current_day=0,
            prices=np.asarray([50.0, 55.0, 300.0]),
            reference_prices=np.asarray([50.0, 55.0, 300.0]),
            inventory=np.asarray([0, 1, 0]),
        )
    finally:
        ABM_CONFIG.utility_noise_std = original_noise

    assert result["requests_by_offset"] == [0, 1, 0]
    assert result["accepted_by_offset"] == [0, 1, 0]
    assert result["rejected_by_capacity"] == [0, 0, 0]


def test_biz_customer_is_rejected_when_first_choice_is_full() -> None:
    original_noise = float(ABM_CONFIG.utility_noise_std)
    ABM_CONFIG.utility_noise_std = 0.0
    try:
        model = _make_single_customer_model("biz")
        result = model.simulate_day(
            current_day=0,
            prices=np.asarray([50.0, 55.0, 300.0]),
            reference_prices=np.asarray([50.0, 55.0, 300.0]),
            inventory=np.asarray([0, 1, 0]),
        )
    finally:
        ABM_CONFIG.utility_noise_std = original_noise

    assert result["requests_by_offset"] == [1, 0, 0]
    assert result["accepted_by_offset"] == [0, 0, 0]
    assert result["rejected_by_capacity"] == [1, 0, 0]
