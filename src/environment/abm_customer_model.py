from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from configs.config import ABM_CONFIG, DATA_CONFIG
from src.utils.preprocess_data import build_demand_calibration, load_filtered_historical_data


class HotelABMModel:
    """`idea2` 的消费者行为模型。

    该版本只保留研究模型需要的要素：
    - 工作日 / 周末非平稳到达
    - 商务型 / 灵活型两类消费者
    - 理想入住日期偏移 d*
    - 支付意愿 WTP
    - 参考价格心理效应
    - 单次到达最多预订一个入住日
    """

    def __init__(self, historical_data: Optional[pd.DataFrame] = None, random_seed: Optional[int] = None):
        self.rng = np.random.default_rng(DATA_CONFIG.seed if random_seed is None else random_seed)
        self.historical_data = (
            load_filtered_historical_data() if historical_data is None else historical_data.copy()
        )
        self.calibration = build_demand_calibration(self.historical_data)
        self.trace_enabled = False
        self.utility_trace: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.utility_trace = []

    @staticmethod
    def is_weekend(day_index: int) -> bool:
        return int(day_index % 7) in (5, 6)

    def get_reference_price_baseline(self, stay_day: int) -> float:
        if self.is_weekend(stay_day):
            return float(self.calibration.weekend_ref_price)
        return float(self.calibration.weekday_ref_price)

    def _sample_arrivals(self, current_day: int) -> int:
        mean_arrivals = (
            self.calibration.weekend_arrival_mean
            if self.is_weekend(current_day)
            else self.calibration.weekday_arrival_mean
        )
        return int(self.rng.poisson(lam=max(0.0, mean_arrivals)))

    def _sample_ideal_offset(self) -> int:
        return int(self.rng.choice(np.arange(3), p=self.calibration.ideal_offset_probs))

    def _sample_customer_type(self) -> str:
        is_flexible = self.rng.random() < float(ABM_CONFIG.flexible_customer_share)
        return "flex" if is_flexible else "biz"

    @staticmethod
    def _get_day_mismatch_penalty(customer_type: str) -> float:
        if customer_type == "biz":
            return float(ABM_CONFIG.lambda_day_mismatch_biz)
        return float(ABM_CONFIG.lambda_day_mismatch_flex)

    def _sample_wtp(self, ideal_offset: int) -> float:
        offset = int(np.clip(ideal_offset, 0, 2))
        mean = float(self.calibration.offset_wtp_means[offset])
        std = float(self.calibration.offset_wtp_stds[offset])
        return float(max(ABM_CONFIG.wtp_min, self.rng.normal(mean, std)))

    def _record_trace(
        self,
        *,
        current_day: int,
        customer_id: int,
        customer_type: str,
        ideal_offset: int,
        wtp: float,
        prices: np.ndarray,
        references: np.ndarray,
        utilities: np.ndarray,
        chosen_offset: Optional[int],
        booked: bool,
    ) -> None:
        if not self.trace_enabled:
            return
        self.utility_trace.append(
            {
                "current_day": int(current_day),
                "customer_id": int(customer_id),
                "customer_type": str(customer_type),
                "ideal_offset": int(ideal_offset),
                "wtp": float(wtp),
                "prices": prices.astype(float).tolist(),
                "references": references.astype(float).tolist(),
                "utilities": utilities.astype(float).tolist(),
                "chosen_offset": None if chosen_offset is None else int(chosen_offset),
                "booked": bool(booked),
            }
        )

    def simulate_day(
        self,
        *,
        current_day: int,
        prices: np.ndarray,
        reference_prices: np.ndarray,
        inventory: np.ndarray,
    ) -> Dict[str, Any]:
        prices = np.asarray(prices, dtype=float).reshape(3)
        reference_prices = np.asarray(reference_prices, dtype=float).reshape(3)
        inventory = np.asarray(inventory, dtype=float).reshape(3)

        arrivals = self._sample_arrivals(current_day)
        requests = np.zeros(3, dtype=int)
        accepted = np.zeros(3, dtype=int)
        rejected_by_capacity = np.zeros(3, dtype=int)
        remaining_inventory = inventory.astype(int).copy()

        for customer_id in range(arrivals):
            customer_type = self._sample_customer_type()
            day_mismatch_penalty = self._get_day_mismatch_penalty(customer_type)
            ideal_offset = self._sample_ideal_offset()
            wtp = self._sample_wtp(ideal_offset)
            noise = self.rng.normal(0.0, ABM_CONFIG.utility_noise_std, size=3)
            utilities = (
                wtp
                - prices
                - day_mismatch_penalty * np.abs(np.arange(3) - ideal_offset)
                + ABM_CONFIG.lambda_reference_price * (reference_prices - prices)
                + noise
            )
            chosen_offset: Optional[int] = None
            booked = False

            if customer_type == "flex":
                feasible_offsets = [int(offset) for offset in np.argsort(utilities)[::-1] if utilities[offset] >= 0.0]
                for offset in feasible_offsets:
                    if remaining_inventory[offset] > 0:
                        chosen_offset = offset
                        booked = True
                        break
                if chosen_offset is None and feasible_offsets:
                    chosen_offset = int(feasible_offsets[0])
            else:
                chosen_offset = int(np.argmax(utilities))
                if utilities[chosen_offset] < 0.0:
                    chosen_offset = None
                elif remaining_inventory[chosen_offset] > 0:
                    booked = True

            if chosen_offset is not None:
                requests[chosen_offset] += 1
                if booked:
                    accepted[chosen_offset] += 1
                    remaining_inventory[chosen_offset] -= 1
                else:
                    rejected_by_capacity[chosen_offset] += 1
                trace_chosen_offset = chosen_offset
            else:
                trace_chosen_offset = None

            self._record_trace(
                current_day=current_day,
                customer_id=customer_id,
                customer_type=customer_type,
                ideal_offset=ideal_offset,
                wtp=wtp,
                prices=prices,
                references=reference_prices,
                utilities=utilities,
                chosen_offset=trace_chosen_offset,
                booked=booked,
            )

        return {
            "arrivals": int(arrivals),
            "requests_by_offset": requests.astype(int).tolist(),
            "accepted_by_offset": accepted.astype(int).tolist(),
            "rejected_by_capacity": rejected_by_capacity.astype(int).tolist(),
        }

    def get_utility_trace(self) -> pd.DataFrame:
        return pd.DataFrame(self.utility_trace)
