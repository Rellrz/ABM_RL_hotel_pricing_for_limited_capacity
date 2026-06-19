"""Oracle Upper Bound：先知双价格上界。

原理：Oracle 偷看 ABM 内部生成的客户信息（WTP、price_sensitivity、lead_time），
对同一 target_date 的客户按类型分别收取统一的市场出清价格：
- online_only 客户 → 统一 pon（线上价），酒店扣除佣金后收入 pon*(1-commission_rate)
- omnichannel 客户 → 统一 poff（线下价），酒店全款收入 poff

定价逻辑与 CEM 完完全全一致：
- 两类客户分别看到各自的价格（pon vs poff）
- 同一类型客户在该 target_date 内支付同一个价格
- OTA 佣金仅在线上渠道产生

Market-clearing 定价规则：将同一 target_date 的客户按 customer_type 拆分，
分别按保留价格降序排列，枚举 (k_on, k_off) 组合（k_on + k_off <= 库存），
选择最大化酒店收入的最优分配：
    收入 = pon * k_on * (1 - commission_rate) + poff * k_off

Oracle 的优势仅在于知道未来客户的确切信息（clairvoyance），
定价结构（两类客户、两类价格）与 CEM 完全对齐，
因此是任何可行策略的严格上界。

保留价格（p*）是使得效用 U(p)=0 的临界价格（anchor=0, noise=0）：
    U(p) = β × (WTP × discount - p) + γ/(lead_time+1)
    p*   = WTP × discount + γ / (β × (lead_time + 1))
其中 discount = online_discount_ratio（online_only）或 1.0（omnichannel）。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs.config import ABM_CONFIG
from configs.experiment2 import Experiment2Config
from src.environment.abm_customer_model import HotelABMModel


def _compute_reservation_price_for_customer(
    customer,
    urgency_weight: float,
    online_discount_ratio: float,
) -> float:
    """计算单个客户的保留价格（无 anchor）。"""
    profile = customer.profile
    beta = float(profile.price_sensitivity)
    wtp = float(profile.wtp)
    lead_time = int(profile.lead_time)

    # 全渠道客户无折扣，仅线上客户有 discount
    if profile.customer_type == "online_only":
        discount = float(online_discount_ratio)
    else:
        discount = 1.0

    # urgency = γ / (L + 1)
    urgency = float(urgency_weight) / float(lead_time + 1)

    # p* = WTP × discount + urgency / β
    if beta < 1e-12:
        return wtp * discount  # 完全价格弹性，WTP 就是上限
    p_star = float(wtp * discount + urgency / beta)
    return max(0.0, p_star)


def _run_single_oracle_episode(
    config: Experiment2Config,
    seed: int,
) -> Tuple[float, float]:
    """跑单个 episode 的 Oracle 上界，返回 (酒店收入, OTA利润)。

    定价逻辑与 CEM 完完全全一致：
    - 同一 target_date 内，online_only 客户统一支付 pon（线上价）
    - 同一 target_date 内，omnichannel 客户统一支付 poff（线下价）
    - 酒店收入 = pon * k_on * (1 - commission_rate) + poff * k_off
    - OTA 利润 = pon * k_on * commission_rate
    - 通过枚举 (k_on, k_off) 组合（满足 k_on + k_off <= 库存），
      选择最大化酒店收入的最优分配。
    """
    total_revenue = 0.0
    total_ota_profit = 0.0
    booking_window = int(config.booking_window_days)
    init_inventory = 226  # ABM 默认每入住日库存
    commission_rate = float(config.commission_rate)

    # 每日可用库存：初始每个 checkin 日均有 init_inventory 间
    daily_available_rooms = np.full(booking_window + 730, init_inventory, dtype=np.int32)

    # 创建 ABM 模型（独立种子）
    abm = HotelABMModel(
        historical_data=pd.DataFrame(),  # Oracle 不需要历史数据，传空 df
        random_seed=seed,
        booking_window_days=booking_window,
    )

    urgency_weight = float(ABM_CONFIG.urgency_weight)
    online_discount_ratio = float(ABM_CONFIG.online_discount_ratio)

    for day in range(730):
        abm.current_day = day
        customers = abm.generate_daily_customers(day)

        # 按 target_date 分组，拆分为 online_only 与 omnichannel 两类
        candidates: Dict[int, Dict[str, List[float]]] = {}
        for c in customers:
            target_date = int(c.profile.target_date)
            if not (day <= target_date < day + booking_window):
                continue
            if daily_available_rooms[target_date] <= 0:
                continue

            p_star = _compute_reservation_price_for_customer(
                c, urgency_weight, online_discount_ratio,
            )
            if p_star <= 0:
                continue

            if target_date not in candidates:
                candidates[target_date] = {"online": [], "offline": []}

            if c.profile.customer_type == "online_only":
                candidates[target_date]["online"].append(p_star)
            else:
                # omnichannel 客户：p* 对应线下价（discount=1.0）
                candidates[target_date]["offline"].append(p_star)

        # 对每个 target_date，分别对两类客户做 market-clearing
        # 枚举 (k_on, k_off) 组合，取最大化酒店收入的分配
        for target_date, groups in candidates.items():
            online_prices = sorted(groups["online"], reverse=True)
            offline_prices = sorted(groups["offline"], reverse=True)

            inventory = int(daily_available_rooms[target_date])

            best_revenue = -1.0
            best_ota = 0.0
            best_k_on = 0
            best_k_off = 0

            n_online = len(online_prices)
            n_offline = len(offline_prices)

            for k_on in range(0, min(n_online, inventory) + 1):
                pon = online_prices[k_on - 1] if k_on > 0 else 0.0
                max_k_off = min(n_offline, inventory - k_on)
                for k_off in range(0, max_k_off + 1):
                    poff = offline_prices[k_off - 1] if k_off > 0 else 0.0

                    # 酒店收入 = 线上收入（扣佣金）+ 线下收入（全款）
                    revenue = pon * k_on * (1.0 - commission_rate) + poff * k_off
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_ota = pon * k_on * commission_rate
                        best_k_on = k_on
                        best_k_off = k_off

            if best_k_on + best_k_off > 0:
                total_revenue += best_revenue
                total_ota_profit += best_ota
                daily_available_rooms[target_date] -= (best_k_on + best_k_off)

        # 每天结束时：当前 day 转为过去，已不可预订
        daily_available_rooms[day] = 0

    return total_revenue, total_ota_profit


def _run_single_seed_oracle(
    config: Experiment2Config,
    historical_data,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """单种子 Oracle 评估（无训练，只有评估）。"""
    del historical_data  # Oracle 不需要历史数据

    train_records: List[Dict] = []  # Oracle 无训练过程
    eval_records: List[Dict] = []
    n_episodes = int(config.post_eval_episodes)

    for ep in range(1, n_episodes + 1):
        ep_seed = seed + ep * 10_000
        hotel_rev, ota_profit = _run_single_oracle_episode(config, ep_seed)
        eval_records.append(
            {
                "Algorithm": "Oracle Upper Bound",
                "Seed": seed,
                "EvalEpisode": ep,
                "EvalHotelRevenue": float(hotel_rev),
                "EvalOTAProfit": float(ota_profit),
                "EvalSystemProfit": float(hotel_rev + ota_profit),
                "EvalRevenue": float(hotel_rev),
            }
        )

    return train_records, eval_records


def run_oracle(
    config: Experiment2Config,
    historical_data,
) -> Tuple[List[Dict], List[Dict]]:
    """Oracle 上界主入口，支持多 seed 并行。

    接口与 run_bo / run_ga 等其他 baseline 一致。
    """
    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []

    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="Oracle Seeds", unit="seed"):
            train_rec, eval_rec = _run_single_seed_oracle(config, historical_data, seed)
            all_train_records.extend(train_rec)
            all_eval_records.extend(eval_rec)
            tqdm.write(f"[Oracle] Seed {seed} done: eval_ep={len(eval_rec)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed_oracle, config, historical_data, seed))

        with tqdm(total=len(futures), desc="Oracle Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                train_rec, eval_rec = fut.result()
                all_train_records.extend(train_rec)
                all_eval_records.extend(eval_rec)
                pbar.update(1)

    return all_train_records, all_eval_records
