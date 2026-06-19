"""实验二共用仿真内核：严格对齐 game_trainer 的分桶训练机制。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.common import (
    build_bucket_mapping,
    compute_bucket_rewards,
    discretize_bucket_state,
    enrich_bucket_state,
    parse_buckets,
)
from configs.experiment2 import Experiment2Config
from src.agent.ota_agent import OTASubsidyHeuristic
from src.environment.hotel_env import HotelEnvironment


@dataclass
class DayResult:
    reward_hotel: float
    reward_ota: float
    done: bool
    info: Dict


@dataclass
class UpdateEvent:
    state: Dict
    action_pair: Tuple[float, float]
    reward: float
    next_state: Dict
    done: bool
    ota_subsidy: float


class BucketPricingSimulator:
    """按分桶价格驱动的一天仿真。

    约定输入动作语义：
    - 每个bucket给出 `(p_online_base, p_offline)`；
    - 线上最终价由 OTA 补贴规则调整后得到。
    """

    def __init__(self, config: Experiment2Config, seed: int, historical_data):
        self.config = config
        self.env = HotelEnvironment(
            initial_inventory=config.initial_inventory,
            historical_data=historical_data,
            booking_window_days=config.booking_window_days,
            episode_days=config.days_per_episode,
        )
        self.ota = OTASubsidyHeuristic(
            commission_rate=config.commission_rate,
            r_max=config.ota_r_max,
            delta_max=config.ota_delta_max,
            decay_lambda=config.ota_decay_lambda,
            noise_std=config.ota_noise_std,
            seed=config.ota_seed if config.ota_seed >= 0 else seed,
        )
        self.buckets = parse_buckets(config.decision_buckets, config.booking_window_days)
        self.bucket_of_offset, self.entry_offsets, self.exit_offsets = build_bucket_mapping(
            self.buckets, config.booking_window_days
        )
        self.day = 0
        self.initialized = False

        self.price_online_base_by_offset: List[float] = []
        self.price_offline_by_offset: List[float] = []
        self.subsidy_ratio_by_offset: List[float] = []
        self.decision_state_by_offset: List[Optional[Dict]] = []
        self.acc_bookings_online_by_offset: List[int] = []
        self.acc_bookings_offline_by_offset: List[int] = []
        self.acc_revenue_online_by_offset: List[float] = []
        self.acc_revenue_offline_by_offset: List[float] = []

    @property
    def n_stages(self) -> int:
        return len(self.buckets)

    def reset(self) -> Dict:
        self.day = 0
        self.initialized = False
        self.price_online_base_by_offset = [0.0] * self.config.booking_window_days
        self.price_offline_by_offset = [0.0] * self.config.booking_window_days
        self.subsidy_ratio_by_offset = [0.0] * self.config.booking_window_days
        self.decision_state_by_offset = [None] * self.config.booking_window_days
        self.acc_bookings_online_by_offset = [0] * self.config.booking_window_days
        self.acc_bookings_offline_by_offset = [0] * self.config.booking_window_days
        self.acc_revenue_online_by_offset = [0.0] * self.config.booking_window_days
        self.acc_revenue_offline_by_offset = [0.0] * self.config.booking_window_days
        self.daily_bookings_per_bucket = [0.0] * self.n_stages
        self.acc_bookings_per_bucket = [0.0] * self.n_stages
        self.acc_bookings_per_bucket_days = 0
        return self.env.reset()

    def _build_stage_raw_state(self, off: int, stage_id: int) -> Dict:
        st = dict(self.env.get_raw_state_for_day_offset(off))
        bucket_start, bucket_end = self.buckets[stage_id]
        st["stage_id"] = int(stage_id)
        st["bucket_start"] = int(bucket_start)
        st["bucket_end"] = int(bucket_end)
        return st

    def get_state_by_stage(self, stage_id: int) -> Dict:
        _s, e = self.buckets[stage_id]
        ref_off = min(e, self.config.booking_window_days - 1)
        return enrich_bucket_state(self._build_stage_raw_state(ref_off, stage_id))

    def get_q_state_by_stage(self, stage_id: int) -> int:
        return discretize_bucket_state(self.get_state_by_stage(stage_id), stage_id=stage_id)

    def get_obs_vector_for_ppo(self) -> np.ndarray:
        st = enrich_bucket_state(self.env.get_raw_state())
        remaining_inventory = float(st.get("inventory_raw", self.config.initial_inventory))
        init_inv = float(max(1.0, self.config.initial_inventory))
        inventory_total = remaining_inventory / init_inv
        inventory_consumed_frac = 1.0 - inventory_total

        future_inventory = np.asarray(
            st.get("future_inventory", [self.config.initial_inventory] * self.config.booking_window_days),
            dtype=np.float64,
        )
        near_i = float(np.sum(future_inventory[0:14])) / max(1.0, len(future_inventory[0:14]) * init_inv)
        far_i = float(np.sum(future_inventory[14:])) / max(1.0, len(future_inventory[14:]) * init_inv)
        inventory_near_available = near_i
        inventory_far_available = far_i

        month = int(((self.day // 30) % 12) + 1)
        month_onehot = np.zeros(12, dtype=np.float64)
        month_onehot[month - 1] = 1.0
        weekend = float(st.get("weekday", 0))
        day_norm = float(self.day % self.config.days_per_episode) / float(max(1, self.config.days_per_episode - 1))

        price_online_min = float(self.config.online_price_min)
        price_online_max = float(self.config.online_price_max)
        price_range_online = max(1e-8, price_online_max - price_online_min)
        price_offline_min = float(self.config.offline_price_min)
        price_offline_max = float(self.config.offline_price_max)
        price_range_offline = max(1e-8, price_offline_max - price_offline_min)

        per_bucket_inventory = np.zeros(self.n_stages, dtype=np.float64)
        per_bucket_online_price = np.zeros(self.n_stages, dtype=np.float64)
        per_bucket_offline_price = np.zeros(self.n_stages, dtype=np.float64)
        per_bucket_online_final = np.zeros(self.n_stages, dtype=np.float64)

        for sid, (s_off, e_off) in enumerate(self.buckets):
            lo = max(0, int(s_off))
            hi = min(int(e_off) + 1, self.config.booking_window_days)
            if lo < hi:
                per_bucket_inventory[sid] = self._sum_offset_range(lo, hi) / init_inv
            ref = min(int(e_off), self.config.booking_window_days - 1)
            pon_val = float(self.price_online_base_by_offset[ref])
            poff_val = float(self.price_offline_by_offset[ref])
            sr_val = float(self.subsidy_ratio_by_offset[ref])
            per_bucket_online_price[sid] = (pon_val - price_online_min) / price_range_online
            per_bucket_offline_price[sid] = (poff_val - price_offline_min) / price_range_offline
            final = pon_val - pon_val * self.config.commission_rate * sr_val
            per_bucket_online_final[sid] = (final - price_online_min) / price_range_online

        recent_bookings = np.asarray(self.acc_bookings_per_bucket, dtype=np.float64)
        recent_bookings = np.clip(recent_bookings / 3.0, 0.0, 1.0)

        vec = np.concatenate(
            [
                np.array([inventory_total, inventory_consumed_frac,
                          inventory_near_available, inventory_far_available], dtype=np.float64),
                month_onehot,
                np.array([weekend, day_norm], dtype=np.float64),
                per_bucket_inventory,
                per_bucket_online_price,
                per_bucket_offline_price,
                per_bucket_online_final,
                recent_bookings,
            ]
        )
        return vec.astype(np.float32)

    def _sum_offset_range(self, lo: int, hi: int) -> float:
        st = enrich_bucket_state(self.env.get_raw_state())
        future_inventory = np.asarray(
            st.get("future_inventory", [self.config.initial_inventory] * self.config.booking_window_days),
            dtype=np.float64,
        )
        return float(np.sum(future_inventory[lo:hi]))

    def _price_clipped(self, action_pair: Tuple[float, float]) -> Tuple[float, float]:
        pon = float(np.clip(action_pair[0], self.config.online_price_min, self.config.online_price_max))
        poff = float(np.clip(action_pair[1], self.config.offline_price_min, self.config.offline_price_max))
        return pon, poff

    def initialize_episode_decisions(self, stage_actions: List[Tuple[float, float]]) -> None:
        """对齐 game_trainer: episode开始时先按每个bucket初始化全窗口决策。"""
        if len(stage_actions) != self.n_stages:
            raise ValueError(f"Expected {self.n_stages} stage actions, got {len(stage_actions)}")
        for sid, (_s, e) in enumerate(self.buckets):
            ref_off = int(min(e, self.config.booking_window_days - 1))
            st = self._build_stage_raw_state(ref_off, sid)
            pon, poff = self._price_clipped(stage_actions[sid])
            sr = float(self.ota.get_subsidy(pon, poff, lead_time=ref_off))
            for off in range(int(_s), min(int(e) + 1, self.config.booking_window_days)):
                self.price_online_base_by_offset[off] = pon
                self.price_offline_by_offset[off] = poff
                self.subsidy_ratio_by_offset[off] = sr
                self.decision_state_by_offset[off] = dict(st)
        self.initialized = True

    def _build_update_event(self, off: int, done_flag: bool) -> Optional[UpdateEvent]:
        bo_acc = int(self.acc_bookings_online_by_offset[off])
        bf_acc = int(self.acc_bookings_offline_by_offset[off])
        if (bo_acc <= 0 and bf_acc <= 0) or self.decision_state_by_offset[off] is None:
            return None

        pon = float(self.price_online_base_by_offset[off])
        poff = float(self.price_offline_by_offset[off])
        sr = float(self.subsidy_ratio_by_offset[off])

        reward_parts = compute_bucket_rewards(
            bookings_online=bo_acc,
            bookings_offline=bf_acc,
            price_online_base=pon,
            price_offline=poff,
            commission_rate=self.config.commission_rate,
            subsidy_ratio=sr,
            reward_hotel_ratio=self.config.reward_hotel_ratio,
            revenue_online=float(self.acc_revenue_online_by_offset[off]),
            revenue_offline=float(self.acc_revenue_offline_by_offset[off]),
            state=self.decision_state_by_offset[off],
            online_price_min=self.config.online_price_min,
            online_price_max=self.config.online_price_max,
            offline_price_min=self.config.offline_price_min,
            offline_price_max=self.config.offline_price_max,
            reward_shape_price_weight=self.config.reward_shape_price_weight,
            reward_shape_sellthrough_weight=self.config.reward_shape_sellthrough_weight,
            reward_shape_target_sellthrough=self.config.reward_shape_target_sellthrough,
        )

        state_for_update = enrich_bucket_state(dict(self.decision_state_by_offset[off]))
        next_state_for_update = enrich_bucket_state(
            self._build_stage_raw_state(off, int(self.bucket_of_offset[off]))
        )
        return UpdateEvent(
            state=state_for_update,
            action_pair=(pon, poff),
            reward=float(reward_parts["reward_hotel"]),
            next_state=next_state_for_update,
            done=bool(done_flag),
            ota_subsidy=float(reward_parts["subsidy_cost"]),
        )

    def _rotate_offsets(self) -> None:
        self.price_online_base_by_offset = self.price_online_base_by_offset[1:] + [self.price_online_base_by_offset[-1]]
        self.price_offline_by_offset = self.price_offline_by_offset[1:] + [self.price_offline_by_offset[-1]]
        self.subsidy_ratio_by_offset = self.subsidy_ratio_by_offset[1:] + [self.subsidy_ratio_by_offset[-1]]
        self.decision_state_by_offset = self.decision_state_by_offset[1:] + [self.decision_state_by_offset[-1]]
        self.acc_bookings_online_by_offset = self.acc_bookings_online_by_offset[1:] + [self.acc_bookings_online_by_offset[-1]]
        self.acc_bookings_offline_by_offset = self.acc_bookings_offline_by_offset[1:] + [self.acc_bookings_offline_by_offset[-1]]
        self.acc_revenue_online_by_offset = self.acc_revenue_online_by_offset[1:] + [self.acc_revenue_online_by_offset[-1]]
        self.acc_revenue_offline_by_offset = self.acc_revenue_offline_by_offset[1:] + [self.acc_revenue_offline_by_offset[-1]]

    def step_day(self, stage_actions: List[Tuple[float, float]]) -> DayResult:
        if len(stage_actions) != self.n_stages:
            raise ValueError(f"Expected {self.n_stages} stage actions, got {len(stage_actions)}")

        if not self.initialized:
            self.initialize_episode_decisions(stage_actions)

        update_events: List[UpdateEvent] = []

        # 在桶的右端点为新进入该桶的cohort重新定价。
        for off in self.entry_offsets:
            sid = int(self.bucket_of_offset[off])
            st = self._build_stage_raw_state(off, sid)
            pon, poff = self._price_clipped(stage_actions[sid])
            sr = float(self.ota.get_subsidy(pon, poff, lead_time=off))
            self.acc_bookings_online_by_offset[off] = 0
            self.acc_bookings_offline_by_offset[off] = 0
            self.acc_revenue_online_by_offset[off] = 0.0
            self.acc_revenue_offline_by_offset[off] = 0.0
            self.price_online_base_by_offset[off] = pon
            self.price_offline_by_offset[off] = poff
            self.subsidy_ratio_by_offset[off] = sr
            self.decision_state_by_offset[off] = dict(st)

        final_online = [
            self.price_online_base_by_offset[i]
            - self.price_online_base_by_offset[i] * self.config.commission_rate * self.subsidy_ratio_by_offset[i]
            for i in range(self.config.booking_window_days)
        ]
        actions_window = [
            [final_online[i], self.price_offline_by_offset[i], self.price_online_base_by_offset[i]]
            for i in range(self.config.booking_window_days)
        ]

        _, _, done, info = self.env.step(actions_window)
        self.day += 1

        bookings = info.get("bookings_by_day_offset", [])
        reward_hotel = 0.0
        reward_ota = 0.0
        by_stage_hotel = [0.0] * self.n_stages
        by_stage_ota = [0.0] * self.n_stages
        for off in range(min(len(bookings), self.config.booking_window_days)):
            bo = int(bookings[off]["bookings_online"])
            bf = int(bookings[off]["bookings_offline"])
            if bo == 0 and bf == 0:
                continue
            sid = int(self.bucket_of_offset[off])
            p_on_base = float(self.price_online_base_by_offset[off])
            p_off = float(self.price_offline_by_offset[off])
            sr = float(self.subsidy_ratio_by_offset[off])
            revenue_online = float(bookings[off].get("revenue_online", 0.0))
            revenue_offline = float(bookings[off].get("revenue_offline", 0.0))
            reward_parts = compute_bucket_rewards(
                bookings_online=bo,
                bookings_offline=bf,
                price_online_base=p_on_base,
                price_offline=p_off,
                commission_rate=self.config.commission_rate,
                subsidy_ratio=sr,
                reward_hotel_ratio=self.config.reward_hotel_ratio,
                revenue_online=revenue_online,
                revenue_offline=revenue_offline,
            )
            reward_hotel += float(reward_parts["revenue_hotel"])
            reward_ota += float(reward_parts["profit_ota"])
            by_stage_hotel[sid] += float(reward_parts["revenue_hotel"])
            by_stage_ota[sid] += float(reward_parts["profit_ota"])
            self.acc_bookings_online_by_offset[off] += int(bo)
            self.acc_bookings_offline_by_offset[off] += int(bf)
            self.acc_revenue_online_by_offset[off] += revenue_online
            self.acc_revenue_offline_by_offset[off] += revenue_offline

        self.daily_bookings_per_bucket = [0.0] * self.n_stages
        for off in range(min(len(bookings), self.config.booking_window_days)):
            bo = int(bookings[off]["bookings_online"])
            bf = int(bookings[off]["bookings_offline"])
            if bo == 0 and bf == 0:
                continue
            sid = int(self.bucket_of_offset[off])
            self.daily_bookings_per_bucket[sid] += float(bo + bf)
        daily_max = 5.0
        for sid in range(self.n_stages):
            self.daily_bookings_per_bucket[sid] = min(1.0, self.daily_bookings_per_bucket[sid] / daily_max)

        self.acc_bookings_per_bucket_days += 1
        for sid in range(self.n_stages):
            self.acc_bookings_per_bucket[sid] += self.daily_bookings_per_bucket[sid]

        if self.acc_bookings_per_bucket_days >= 7:
            for sid in range(self.n_stages):
                self.acc_bookings_per_bucket[sid] *= 0.7
            self.acc_bookings_per_bucket_days = 0

        done = bool(done or self.day >= self.config.days_per_episode)

        # 在桶的左端点结算该cohort完整经历该桶后的累计收益。
        for off in self.exit_offsets:
            ev = self._build_update_event(off, done_flag=done)
            if ev is not None:
                update_events.append(ev)
            self.acc_bookings_online_by_offset[off] = 0
            self.acc_bookings_offline_by_offset[off] = 0
            self.acc_revenue_online_by_offset[off] = 0.0
            self.acc_revenue_offline_by_offset[off] = 0.0
            self.decision_state_by_offset[off] = None

        if done:
            # 对齐 game_trainer：episode结束后flush全部offset累计收益（done=True）
            for off in range(self.config.booking_window_days):
                ev = self._build_update_event(off, done_flag=True)
                if ev is not None:
                    update_events.append(ev)
                self.acc_bookings_online_by_offset[off] = 0
                self.acc_bookings_offline_by_offset[off] = 0
                self.acc_revenue_online_by_offset[off] = 0.0
                self.acc_revenue_offline_by_offset[off] = 0.0
        else:
            self._rotate_offsets()

        shaped_bucket_reward = 0.0
        shaped_bucket_reward_by_stage = [0.0] * self.n_stages
        for ev in update_events:
            ev_reward = float(ev.reward)
            shaped_bucket_reward += ev_reward
            sid = int(ev.state.get("stage_id", 0)) if isinstance(ev.state, dict) else 0
            if 0 <= sid < self.n_stages:
                shaped_bucket_reward_by_stage[sid] += ev_reward

        info = dict(info)
        info["reward_hotel_by_stage"] = by_stage_hotel
        info["reward_ota_by_stage"] = by_stage_ota
        info["ppo_shaped_bucket_reward"] = float(shaped_bucket_reward)
        info["ppo_shaped_bucket_reward_by_stage"] = shaped_bucket_reward_by_stage
        info["update_events"] = update_events
        return DayResult(
            reward_hotel=float(reward_hotel),
            reward_ota=float(reward_ota),
            done=done,
            info=info,
        )
