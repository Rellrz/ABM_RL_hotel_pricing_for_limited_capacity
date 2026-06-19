"""项目通用工具（状态离散化、分桶映射、奖励计算等）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def parse_buckets(spec: str, n: int) -> List[Tuple[int, int]]:
    tokens = [t.strip() for t in str(spec).replace(",", "|").split("|") if t.strip()]
    buckets: List[Tuple[int, int]] = []
    for token in tokens:
        if "-" in token:
            a, b = token.split("-", 1)
            s, e = int(a), int(b)
        else:
            s = e = int(token)
        buckets.append((s, e))
    buckets.sort(key=lambda x: x[0])

    if not buckets:
        raise ValueError("decision_buckets cannot be empty")
    if buckets[0][0] != 0 or buckets[-1][1] != n - 1:
        raise ValueError(f"Buckets must cover [0, {n-1}]")

    prev_end = -1
    for s, e in buckets:
        if s != prev_end + 1 or e < s:
            raise ValueError("Buckets must be contiguous and valid")
        prev_end = e
    return buckets


def build_bucket_mapping(buckets: List[Tuple[int, int]], window_days: int) -> Tuple[List[int], List[int], List[int]]:
    bucket_of_offset = [0] * window_days
    for sid, (s, e) in enumerate(buckets):
        for off in range(s, e + 1):
            bucket_of_offset[off] = sid
    entry_offsets = sorted({e for _, e in buckets if 0 <= e < window_days})
    exit_offsets = sorted({s for s, _ in buckets if 0 <= s < window_days})
    return bucket_of_offset, entry_offsets, exit_offsets


N_INVENTORY_LEVELS = 5
N_SEASONS = 3
N_WEEKDAY_TYPES = 2
N_STAGE_BUCKETS = 8
BASE_STATE_COUNT = N_INVENTORY_LEVELS * N_SEASONS * N_WEEKDAY_TYPES
TOTAL_Q_STATES = BASE_STATE_COUNT * N_STAGE_BUCKETS


def season_from_day(day: int) -> int:
    month = (int(day) // 30) % 12 + 1
    if month in (11, 12, 1, 2):
        return 0
    if month in (6, 7, 8):
        return 2
    return 1


def weekday_type_from_day(day: int) -> int:
    return 1 if (int(day) % 7) in (5, 6) else 0


def discretize_inventory_from_raw(
    inventory_raw: float,
    initial_inventory: float,
    n_inventory_levels: int = N_INVENTORY_LEVELS,
) -> int:
    inv = float(inventory_raw)
    init_inv = float(max(1.0, initial_inventory))
    if n_inventory_levels <= 1:
        return 0
    ratio = float(np.clip(inv / init_inv, 0.0, 1.0))
    # 5档默认阈值: 0.2 / 0.4 / 0.6 / 0.8
    if n_inventory_levels == 5:
        if ratio <= 0.2:
            return 0
        if ratio <= 0.4:
            return 1
        if ratio <= 0.6:
            return 2
        if ratio <= 0.8:
            return 3
        return 4
    level = int(np.floor(ratio * n_inventory_levels))
    return int(np.clip(level, 0, n_inventory_levels - 1))


def _mean_ratio(values: List[float], initial_inventory: float) -> float:
    if not values:
        return 1.0
    init_inv = float(max(1.0, initial_inventory))
    return float(np.mean(np.asarray(values, dtype=float)) / init_inv)


def _clip_ratio_bin(ratio: float, n_inventory_levels: int = N_INVENTORY_LEVELS) -> int:
    return int(
        discretize_inventory_from_raw(
            inventory_raw=float(np.clip(ratio, 0.0, 1.0)),
            initial_inventory=1.0,
            n_inventory_levels=n_inventory_levels,
        )
    )


def enrich_bucket_state(
    state: Dict,
    n_inventory_levels: int = N_INVENTORY_LEVELS,
) -> Dict:
    """将环境原始状态补齐为离散策略所需状态字段。"""
    out = dict(state)
    day = int(out.get("day", 0))
    init_inv = float(out.get("initial_inventory", max(1.0, out.get("inventory_raw", 1.0))))
    if "season" not in out:
        out["season"] = int(season_from_day(day))
    if "weekday" not in out:
        out["weekday"] = int(weekday_type_from_day(day))
    if "inventory_level" not in out:
        inv_raw = float(out.get("inventory_raw", 0.0))
        out["inventory_level"] = int(
            discretize_inventory_from_raw(
                inventory_raw=inv_raw,
                initial_inventory=init_inv,
                n_inventory_levels=n_inventory_levels,
            )
        )
    future_inventory = list(out.get("future_inventory", []) or [])
    bucket_end = int(out.get("bucket_end", out.get("day_offset", 0)))
    if future_inventory:
        last_idx = len(future_inventory) - 1
        bucket_end = int(np.clip(bucket_end, 0, last_idx))
    else:
        bucket_end = 0

    fallback_slice = [out.get("inventory_raw", init_inv)]
    near_slice = future_inventory[: min(7, len(future_inventory))] if future_inventory else fallback_slice
    far_anchor = min(max(30, bucket_end + 1), len(future_inventory) - 1) if future_inventory else 0
    far_slice = (
        future_inventory[far_anchor:]
        if future_inventory and far_anchor < len(future_inventory)
        else future_inventory[-min(30, len(future_inventory)) :]
        if future_inventory
        else fallback_slice
    )

    near_inv_ratio = _mean_ratio(near_slice, initial_inventory=init_inv)
    far_inv_ratio = _mean_ratio(far_slice, initial_inventory=init_inv)

    out["near_inv_ratio"] = float(near_inv_ratio)
    out["far_inv_ratio"] = float(far_inv_ratio)
    out["near_inv_bin"] = int(_clip_ratio_bin(near_inv_ratio, n_inventory_levels=n_inventory_levels))
    out["far_inv_bin"] = int(_clip_ratio_bin(far_inv_ratio, n_inventory_levels=n_inventory_levels))
    return out


def discretize_bucket_state(
    state: Dict,
    stage_id: int,
    n_inventory_levels: int = N_INVENTORY_LEVELS,
    n_seasons: int = N_SEASONS,
    n_weekday_types: int = N_WEEKDAY_TYPES,
    n_stage_buckets: int = N_STAGE_BUCKETS,
) -> int:
    """统一的CEM/Q状态离散函数（库存×季节×周末×bucket）。"""
    norm = enrich_bucket_state(state, n_inventory_levels=n_inventory_levels)
    inv = int(np.clip(int(norm.get("inventory_level", n_inventory_levels - 1)), 0, n_inventory_levels - 1))
    season = int(np.clip(int(norm.get("season", 0)), 0, n_seasons - 1))
    weekday = int(np.clip(int(norm.get("weekday", 0)), 0, n_weekday_types - 1))
    stage_id = int(np.clip(stage_id, 0, n_stage_buckets - 1))
    base_state = inv * (n_seasons * n_weekday_types) + season * n_weekday_types + weekday
    return int(base_state * n_stage_buckets + stage_id)


def state_to_q_state(state: Dict, stage_id: int) -> int:
    return discretize_bucket_state(state, stage_id=stage_id)


def state_to_144(state: Dict, stage_id: int) -> int:
    """Backward-compatible alias. The state space now has 240 states."""
    return discretize_bucket_state(state, stage_id=stage_id)


def build_cem_state_key(
    state: Dict,
    stage_id: int | None = None,
) -> Tuple[int, ...]:
    """为CEM构造固定的5维状态键。"""
    norm = enrich_bucket_state(state)
    stage = int(norm.get("stage_id", 0) if stage_id is None else stage_id)
    return (
        stage,
        int(norm.get("season", 0)),
        int(norm.get("weekday", 0)),
        int(norm.get("near_inv_bin", norm.get("inventory_level", N_INVENTORY_LEVELS - 1))),
        int(norm.get("far_inv_bin", norm.get("inventory_level", N_INVENTORY_LEVELS - 1))),
    )


def build_cem_flat_state(state: Dict, stage_id: int | None = None) -> int:
    """build_cem_state_key → 扁平整数索引（0~1199），供 Q-learning 等表格方法复用 CEM 状态划分。"""
    stage, season, weekday, near_bin, far_bin = build_cem_state_key(state, stage_id)
    return (((stage * 3 + season) * 2 + weekday) * 5 + near_bin) * 5 + far_bin


def compute_reward_shaping(
    state: Dict | None,
    base_reward_hotel: float,
    bookings_online: int,
    bookings_offline: int,
    price_online_base: float,
    price_offline: float,
    online_price_min: float,
    online_price_max: float,
    offline_price_min: float,
    offline_price_max: float,
    price_weight: float,
    sellthrough_weight: float,
    target_sellthrough: float,
) -> Dict[str, float]:
    """V2 reward shaping：加性机会成本惩罚，用于改变 CEM 样本排序。"""
    if state is None or (price_weight <= 0.0 and sellthrough_weight <= 0.0):
        return {
            "pressure": 0.0,
            "low_price_signal": 0.0,
            "sellthrough": 0.0,
            "sellthrough_excess": 0.0,
            "price_penalty_ratio": 0.0,
            "sellthrough_penalty_ratio": 0.0,
            "shaping_penalty_ratio": 0.0,
            "price_penalty": 0.0,
            "sellthrough_penalty": 0.0,
            "shaping_penalty": 0.0,
            "shaping_penalty_amount": 0.0,
            "reward_multiplier": 1.0,
        }

    norm = enrich_bucket_state(state)
    inventory_ratio = float(np.clip(norm.get("inventory_ratio", 1.0), 0.0, 1.0))
    near_inv_ratio = float(np.clip(norm.get("near_inv_ratio", inventory_ratio), 0.0, 1.0))
    far_inv_ratio = float(np.clip(norm.get("far_inv_ratio", inventory_ratio), 0.0, 1.0))

    scarcity = float(np.clip(1.0 - near_inv_ratio, 0.0, 1.0))
    near_tightness = float(np.clip(far_inv_ratio - near_inv_ratio, 0.0, 1.0))
    pressure = float(np.clip(0.6 * scarcity + 0.4 * near_tightness, 0.0, 1.0))

    on_span = float(max(1e-8, online_price_max - online_price_min))
    off_span = float(max(1e-8, offline_price_max - offline_price_min))
    on_pos = float(np.clip((price_online_base - online_price_min) / on_span, 0.0, 1.0))
    off_pos = float(np.clip((price_offline - offline_price_min) / off_span, 0.0, 1.0))
    avg_price_pos = 0.5 * (on_pos + off_pos)
    low_price_signal = float(np.clip(1.0 - avg_price_pos, 0.0, 1.0))

    inventory_ref = float(max(1.0, norm.get("inventory_raw", norm.get("initial_inventory", 1.0))))
    sellthrough = float(np.clip((int(bookings_online) + int(bookings_offline)) / inventory_ref, 0.0, 1.0))
    sellthrough_excess = float(max(0.0, sellthrough - float(np.clip(target_sellthrough, 0.0, 1.0))))

    # 非线性放大高压低价、过快售出的坏样本，提升对 CEM 排序的影响力。
    price_penalty_ratio = float(max(0.0, price_weight) * (pressure ** 1.4) * (low_price_signal ** 1.25))
    sellthrough_penalty_ratio = float(
        max(0.0, sellthrough_weight) * (pressure ** 1.2) * (sellthrough_excess ** 1.15)
    )
    shaping_penalty_ratio = float(np.clip(price_penalty_ratio + sellthrough_penalty_ratio, 0.0, 0.45))
    scale = float(max(1.0, base_reward_hotel))
    price_penalty = float(scale * price_penalty_ratio)
    sellthrough_penalty = float(scale * sellthrough_penalty_ratio)
    shaping_penalty_amount = float(scale * shaping_penalty_ratio)
    reward_multiplier = float(max(0.0, 1.0 - shaping_penalty_ratio))

    return {
        "pressure": pressure,
        "low_price_signal": low_price_signal,
        "sellthrough": sellthrough,
        "sellthrough_excess": sellthrough_excess,
        "price_penalty_ratio": price_penalty_ratio,
        "sellthrough_penalty_ratio": sellthrough_penalty_ratio,
        "shaping_penalty_ratio": shaping_penalty_ratio,
        "price_penalty": price_penalty,
        "sellthrough_penalty": sellthrough_penalty,
        "shaping_penalty": shaping_penalty_ratio,
        "shaping_penalty_amount": shaping_penalty_amount,
        "reward_multiplier": reward_multiplier,
    }


def compute_bucket_rewards(
    bookings_online: int,
    bookings_offline: int,
    price_online_base: float,
    price_offline: float,
    commission_rate: float,
    subsidy_ratio: float,
    reward_hotel_ratio: float,
    revenue_online: float | None = None,
    revenue_offline: float | None = None,
    state: Dict | None = None,
    online_price_min: float | None = None,
    online_price_max: float | None = None,
    offline_price_min: float | None = None,
    offline_price_max: float | None = None,
    reward_shape_price_weight: float = 0.0,
    reward_shape_sellthrough_weight: float = 0.0,
    reward_shape_target_sellthrough: float = 0.30,
) -> Dict[str, float]:
    """统一CEM奖励口径：酒店收益、OTA利润、系统收益与训练奖励。"""
    bo = int(max(0, bookings_online))
    bf = int(max(0, bookings_offline))
    pon = float(price_online_base)
    poff = float(price_offline)
    c = float(commission_rate)
    sr = float(np.clip(subsidy_ratio, 0.0, 1.0))
    r_h = float(np.clip(reward_hotel_ratio, 0.0, 1.0))

    revenue_online_hotel = float(bo * pon * (1.0 - c) if revenue_online is None else revenue_online)
    revenue_offline_hotel = float(bf * poff if revenue_offline is None else revenue_offline)
    revenue_hotel = revenue_online_hotel + revenue_offline_hotel
    commission_revenue = bo * pon * c
    subsidy_cost = commission_revenue * sr
    profit_ota = commission_revenue - subsidy_cost
    system_profit = revenue_hotel + profit_ota
    base_reward_hotel = r_h * revenue_hotel + (1.0 - r_h) * system_profit

    shaping_parts = compute_reward_shaping(
        state=state,
        base_reward_hotel=base_reward_hotel,
        bookings_online=bo,
        bookings_offline=bf,
        price_online_base=pon,
        price_offline=poff,
        online_price_min=float(pon if online_price_min is None else online_price_min),
        online_price_max=float(pon if online_price_max is None else online_price_max),
        offline_price_min=float(poff if offline_price_min is None else offline_price_min),
        offline_price_max=float(poff if offline_price_max is None else offline_price_max),
        price_weight=float(reward_shape_price_weight),
        sellthrough_weight=float(reward_shape_sellthrough_weight),
        target_sellthrough=float(reward_shape_target_sellthrough),
    )
    reward_hotel = float(max(0.0, base_reward_hotel - shaping_parts["shaping_penalty_amount"]))

    return {
        "revenue_hotel": float(revenue_hotel),
        "profit_ota": float(profit_ota),
        "system_profit": float(system_profit),
        "base_reward_hotel": float(base_reward_hotel),
        "reward_hotel": float(reward_hotel),
        "subsidy_cost": float(subsidy_cost),
        "commission_revenue": float(commission_revenue),
        **shaping_parts,
    }


def q_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return float(eps_end)
    ratio = 1.0 - float(step) / float(decay_steps)
    return float(eps_end + (eps_start - eps_end) * ratio)
