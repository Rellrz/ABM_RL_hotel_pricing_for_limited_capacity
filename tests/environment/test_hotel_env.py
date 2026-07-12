from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline.pricing_baselines import get_inventory_protection_policy, get_weekday_weekend_static_policy
from src.environment.hotel_env import HotelEnvironment


def make_env(start_day: int) -> HotelEnvironment:
    env = HotelEnvironment(historical_data=pd.DataFrame(), random_seed=42, capacity=40)
    env.start_day = int(start_day)
    env.reset()
    return env


def test_state_vector_exposes_weekday_flags_for_each_priced_day() -> None:
    friday_env = make_env(start_day=4)
    saturday_env = make_env(start_day=5)

    assert friday_env.get_state_vector()[:3].tolist() == [1.0, 0.0, 0.0]
    assert saturday_env.get_state_vector()[:3].tolist() == [0.0, 0.0, 1.0]


def test_state_vector_keeps_inventory_and_reference_price_offsets_after_date_flags() -> None:
    env = make_env(start_day=4)
    state = env.get_state_vector()

    assert state[3:6].tolist() == [40.0, 40.0, 40.0]
    assert np.allclose(state[6:9], env.reference_price_window.astype(np.float32))


def test_weekday_weekend_static_policy_uses_each_offset_date_flag() -> None:
    policy = get_weekday_weekend_static_policy(
        weekday_prices=(100.0, 110.0, 120.0),
        weekend_prices=(200.0, 210.0, 220.0),
    )
    obs = np.asarray([1.0, 0.0, 0.0, 40.0, 40.0, 40.0, 100.0, 100.0, 100.0], dtype=np.float32)

    assert policy(obs).tolist() == [100.0, 210.0, 220.0]


def test_inventory_protection_policy_reads_inventory_after_date_flags() -> None:
    policy = get_inventory_protection_policy(base_prices=(100.0, 100.0, 100.0), scarcity_alpha=40.0, capacity=40)
    obs = np.asarray([1.0, 0.0, 0.0, 10.0, 20.0, 40.0, 100.0, 100.0, 100.0], dtype=np.float32)

    assert policy(obs).tolist() == [130.0, 120.0, 100.0]
