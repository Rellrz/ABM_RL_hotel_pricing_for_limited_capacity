from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from configs.config import ABM_CONFIG, ENV_CONFIG
from src.environment.abm_customer_model import HotelABMModel


class HotelEnvironment:
    """`idea2` 的最小可运行酒店定价环境。"""

    def __init__(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        random_seed: Optional[int] = None,
        capacity: Optional[int] = None,
        variable_cost_per_room: Optional[float] = None,
        scarcity_threshold_ratio: Optional[float] = None,
        scarcity_penalty_coef: Optional[float] = None,
        scarcity_penalty_weights: Optional[tuple[float, float, float] | list[float]] = None,
    ):
        self.capacity = int(ENV_CONFIG.capacity if capacity is None else capacity)
        self.window_size = int(ABM_CONFIG.window_size)
        self.episode_days = int(ENV_CONFIG.episode_days)
        self.price_min = float(ENV_CONFIG.price_min)
        self.price_max = float(ENV_CONFIG.price_max)
        self.variable_cost_per_room = float(
            ENV_CONFIG.variable_cost_per_room if variable_cost_per_room is None else variable_cost_per_room
        )
        if self.variable_cost_per_room < 0.0:
            raise ValueError("variable_cost_per_room 不能为负数。")
        self.scarcity_threshold_ratio = float(
            ENV_CONFIG.scarcity_threshold_ratio if scarcity_threshold_ratio is None else scarcity_threshold_ratio
        )
        self.scarcity_penalty_coef = float(
            ENV_CONFIG.scarcity_penalty_coef if scarcity_penalty_coef is None else scarcity_penalty_coef
        )
        weights = ENV_CONFIG.scarcity_penalty_weights if scarcity_penalty_weights is None else scarcity_penalty_weights
        self.scarcity_penalty_weights = np.asarray(weights, dtype=np.float64).reshape(self.window_size)
        if np.any(self.scarcity_penalty_weights < 0.0):
            raise ValueError("scarcity_penalty_weights 不能包含负数。")
        self.start_day = int(ENV_CONFIG.start_day)
        self.abm_model = HotelABMModel(historical_data=historical_data, random_seed=random_seed)
        self.current_day = int(self.start_day)
        self.inventory_window = np.full(self.window_size, self.capacity, dtype=np.int32)
        self.reference_price_window = np.zeros(self.window_size, dtype=np.float64)
        self.total_reward = 0.0
        self.total_revenue = 0.0
        self.daily_history: list[dict[str, Any]] = []
        self.reset()

    def _day_index(self, day: int) -> int:
        return int(day % 7) + 1

    def _is_weekend(self, day: int) -> int:
        return 1 if (day % 7) in (5, 6) else 0

    def _is_weekday(self, day: int) -> int:
        return 0 if self._is_weekend(day) else 1

    def _window_is_weekday(self, base_day: int) -> np.ndarray:
        return np.asarray(
            [self._is_weekday(base_day + offset) for offset in range(self.window_size)],
            dtype=np.float64,
        )

    def _initial_reference_window(self, base_day: int) -> np.ndarray:
        return np.asarray(
            [self.abm_model.get_reference_price_baseline(base_day + offset) for offset in range(self.window_size)],
            dtype=np.float64,
        )

    def _clip_prices(self, action: Any) -> np.ndarray:
        prices = np.asarray(action, dtype=np.float64).reshape(self.window_size)
        return np.clip(prices, self.price_min, self.price_max)

    def _compute_reward_components(self, revenue: float, inventory_after: np.ndarray) -> tuple[float, float]:
        remaining_ratio = inventory_after.astype(np.float64) / float(self.capacity)
        scarcity_gap = np.maximum(self.scarcity_threshold_ratio - remaining_ratio, 0.0)
        scarcity_penalty = float(
            self.scarcity_penalty_coef * np.sum(self.scarcity_penalty_weights * scarcity_gap**2)
        )
        reward = float(revenue - scarcity_penalty)
        return reward, scarcity_penalty

    def _build_state(self) -> Dict[str, Any]:
        return {
            "weekday_index": self._day_index(self.current_day),
            "is_weekend": self._is_weekend(self.current_day),
            "is_weekday_by_offset": self._window_is_weekday(self.current_day),
            "inventory": self.inventory_window.astype(np.float64).copy(),
            "reference_prices": self.reference_price_window.astype(np.float64).copy(),
            "day_mod_7": int(self.current_day % 7),
            "day": int(self.current_day),
        }

    def get_state_vector(self) -> np.ndarray:
        state = self._build_state()
        return np.asarray(
            [
                float(state["is_weekday_by_offset"][0]),
                float(state["is_weekday_by_offset"][1]),
                float(state["is_weekday_by_offset"][2]),
                float(state["inventory"][0]),
                float(state["inventory"][1]),
                float(state["inventory"][2]),
                float(state["reference_prices"][0]),
                float(state["reference_prices"][1]),
                float(state["reference_prices"][2]),
            ],
            dtype=np.float32,
        )

    def reset(self) -> Dict[str, Any]:
        self.current_day = int(self.start_day)
        self.inventory_window = np.full(self.window_size, self.capacity, dtype=np.int32)
        self.reference_price_window = self._initial_reference_window(self.current_day)
        self.total_reward = 0.0
        self.total_revenue = 0.0
        self.daily_history = []
        self.abm_model.reset()
        return self._build_state()

    def step(self, action: Any) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        prices = self._clip_prices(action)
        inventory_before = self.inventory_window.astype(np.int32).copy()
        reference_before = self.reference_price_window.astype(np.float64).copy()

        demand = self.abm_model.simulate_day(
            current_day=self.current_day,
            prices=prices,
            reference_prices=reference_before,
            inventory=inventory_before,
        )

        requests = np.asarray(demand["requests_by_offset"], dtype=np.int32)
        accepted = np.asarray(demand["accepted_by_offset"], dtype=np.int32)
        inventory_after = np.maximum(inventory_before - accepted, 0)
        gross_revenue_by_offset = prices * accepted.astype(np.float64)
        variable_cost_by_offset = self.variable_cost_per_room * accepted.astype(np.float64)
        revenue_by_offset = gross_revenue_by_offset - variable_cost_by_offset
        gross_revenue = float(np.sum(gross_revenue_by_offset))
        variable_cost = float(np.sum(variable_cost_by_offset))
        revenue = float(np.sum(revenue_by_offset))
        remaining_ratio = inventory_after.astype(np.float64) / float(self.capacity)
        reward, scarcity_penalty = self._compute_reward_components(
            revenue=revenue,
            inventory_after=inventory_after,
        )
        total_penalty = float(scarcity_penalty)

        updated_reference = (
            ABM_CONFIG.reference_memory_alpha * reference_before
            + (1.0 - ABM_CONFIG.reference_memory_alpha) * prices
        )

        self.total_reward += reward
        self.total_revenue += revenue

        history_row = {
            "day": int(self.current_day),
            "weekday_index": int(self._day_index(self.current_day)),
            "is_weekend": int(self._is_weekend(self.current_day)),
            "arrivals": int(demand["arrivals"]),
            "arrivals_by_ideal_offset": demand["arrivals_by_ideal_offset"],
            "prices": prices.astype(float).tolist(),
            "reference_prices_before": reference_before.astype(float).tolist(),
            "requests_by_offset": requests.astype(int).tolist(),
            "accepted_by_offset": accepted.astype(int).tolist(),
            "inventory_before": inventory_before.astype(int).tolist(),
            "inventory_after": inventory_after.astype(int).tolist(),
            "remaining_ratio": remaining_ratio.astype(float).tolist(),
            "gross_revenue_by_offset": gross_revenue_by_offset.astype(float).tolist(),
            "variable_cost_by_offset": variable_cost_by_offset.astype(float).tolist(),
            "revenue_by_offset": revenue_by_offset.astype(float).tolist(),
            "gross_revenue": float(gross_revenue),
            "variable_cost": float(variable_cost),
            "variable_cost_per_room": float(self.variable_cost_per_room),
            "revenue": float(revenue),
            "scarcity_penalty": float(scarcity_penalty),
            "total_penalty": float(total_penalty),
            "reward": float(reward),
        }
        self.daily_history.append(history_row)

        next_inventory = np.asarray(
            [inventory_after[1], inventory_after[2], self.capacity],
            dtype=np.int32,
        )
        next_reference = np.asarray(
            [
                updated_reference[1],
                updated_reference[2],
                self.abm_model.get_reference_price_baseline(self.current_day + self.window_size),
            ],
            dtype=np.float64,
        )

        self.current_day += 1
        self.inventory_window = next_inventory
        self.reference_price_window = next_reference

        done = bool((self.current_day - self.start_day) >= self.episode_days)
        next_state = self._build_state()
        info = {
            "arrivals": int(demand["arrivals"]),
            "arrivals_by_ideal_offset": demand["arrivals_by_ideal_offset"],
            "prices": prices.astype(float).tolist(),
            "reference_prices_before": reference_before.astype(float).tolist(),
            "reference_prices_after_update": updated_reference.astype(float).tolist(),
            "requests_by_offset": requests.astype(int).tolist(),
            "accepted_by_offset": accepted.astype(int).tolist(),
            "rejected_by_capacity": demand["rejected_by_capacity"],
            "inventory_before": inventory_before.astype(int).tolist(),
            "inventory_after": inventory_after.astype(int).tolist(),
            "remaining_ratio": remaining_ratio.astype(float).tolist(),
            "rolled_inventory": next_inventory.astype(int).tolist(),
            "gross_revenue_by_offset": gross_revenue_by_offset.astype(float).tolist(),
            "variable_cost_by_offset": variable_cost_by_offset.astype(float).tolist(),
            "revenue_by_offset": revenue_by_offset.astype(float).tolist(),
            "gross_revenue": float(gross_revenue),
            "variable_cost": float(variable_cost),
            "variable_cost_per_room": float(self.variable_cost_per_room),
            "revenue": float(revenue),
            "scarcity_penalty": float(scarcity_penalty),
            "total_penalty": float(total_penalty),
            "reward": float(reward),
        }
        return next_state, reward, done, info

    def get_statistics(self) -> Dict[str, float]:
        if not self.daily_history:
            return {}
        history = pd.DataFrame(self.daily_history)
        arrivals_total = float(history["arrivals"].sum())
        accepted_total = float(np.sum(np.vstack(history["accepted_by_offset"].to_list())))
        full_events = float(np.sum(np.asarray(history["inventory_after"].to_list()) == 0))
        return {
            "total_days": float(len(history)),
            "total_reward": float(self.total_reward),
            "total_revenue": float(self.total_revenue),
            "avg_daily_revenue": float(self.total_revenue / max(1, len(history))),
            "avg_acceptance_rate": float(accepted_total / max(1.0, arrivals_total)),
            "full_capacity_events": float(full_events),
        }
