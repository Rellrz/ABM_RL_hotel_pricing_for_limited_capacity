"""EMSR-b baseline runner（静态公式，无训练过程）。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import log, sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.common import parse_buckets
from configs.experiment2 import Experiment2Config
from src.evaluation.policy_evaluator import evaluate_policy


@dataclass
class EMSRbProfile:
    global_adr: float
    season_weekday_factor: Dict[Tuple[int, int], float]
    stage_price_factor: List[float]
    stage_online_share: List[float]


def _norm_ppf(p: float) -> float:
    """Acklam 近似：标准正态分布逆CDF，避免额外依赖 scipy。"""
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = sqrt(-2.0 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _month_to_int(month_val) -> int:
    if isinstance(month_val, (int, np.integer)):
        return int(np.clip(int(month_val), 1, 12))
    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    return int(month_map.get(str(month_val), 1))


def _season_from_month(month: int) -> int:
    if month in (11, 12, 1, 2):
        return 0
    if month in (6, 7, 8):
        return 2
    return 1


def _prepare_historical_frame(historical_data: pd.DataFrame, booking_window_days: int) -> pd.DataFrame:
    df = historical_data.copy()
    if "is_canceled" in df.columns:
        df = df[df["is_canceled"] == 0].copy()
    if "adr" in df.columns:
        df = df[df["adr"] > 0].copy()

    df["lead_time_clip"] = df["lead_time"].fillna(0).astype(int).clip(0, booking_window_days - 1)
    df["month_int"] = df["arrival_date_month"].map(_month_to_int).astype(int)
    df["season"] = df["month_int"].map(_season_from_month).astype(int)

    arr_date = pd.to_datetime(
        dict(
            year=df["arrival_date_year"].astype(int),
            month=df["month_int"].astype(int),
            day=df["arrival_date_day_of_month"].astype(int),
        ),
        errors="coerce",
    )
    weekday = arr_date.dt.weekday.fillna(0).astype(int)
    df["weekday"] = np.where(weekday >= 5, 1, 0).astype(int)

    online_mask = (
        (df["market_segment"].astype(str) == "Online TA")
        | (df["distribution_channel"].astype(str).isin(["TA/TO", "GDS"]))
    )
    df["is_online"] = online_mask.astype(int)
    df["arrival_date"] = arr_date.dt.date.astype(str)
    return df


def _build_emsrb_profile(config: Experiment2Config, historical_data: pd.DataFrame) -> EMSRbProfile:
    df = _prepare_historical_frame(historical_data, config.booking_window_days)
    buckets = parse_buckets(config.decision_buckets, config.booking_window_days)

    global_adr = float(df["adr"].median()) if len(df) > 0 else float((config.online_price_min + config.offline_price_max) * 0.5)
    global_adr = float(np.clip(global_adr, config.online_price_min, config.offline_price_max))

    sw = df.groupby(["season", "weekday"])["adr"].mean() if len(df) > 0 else pd.Series(dtype=float)
    season_weekday_factor: Dict[Tuple[int, int], float] = {}
    for season in (0, 1, 2):
        for weekday in (0, 1):
            key = (season, weekday)
            if key in sw.index:
                fac = float(sw.loc[key] / max(global_adr, 1e-6))
            else:
                fac = 1.0
            season_weekday_factor[key] = float(np.clip(fac, 0.75, 1.30))

    mus: List[float] = []
    sigmas: List[float] = []
    fares: List[float] = []
    online_share: List[float] = []
    n_arrival_days = max(1, int(df["arrival_date"].nunique())) if len(df) > 0 else 1
    for (s, e) in buckets:
        stage_df = df[(df["lead_time_clip"] >= int(s)) & (df["lead_time_clip"] <= int(e))].copy()
        if len(stage_df) == 0:
            mu = 0.0
            sigma = 1.0
            fare = global_adr
            on_share = 0.5
        else:
            per_day = stage_df.groupby("arrival_date").size()
            mu = float(per_day.mean()) if len(per_day) > 0 else float(len(stage_df) / n_arrival_days)
            sigma = float(per_day.std(ddof=0)) if len(per_day) > 1 else max(1.0, sqrt(max(mu, 1e-6)))
            fare = float(stage_df["adr"].median())
            on_share = float(stage_df["is_online"].mean())
        mus.append(float(max(mu, 1e-6)))
        sigmas.append(float(max(sigma, 1e-6)))
        fares.append(float(np.clip(fare, config.online_price_min, config.offline_price_max)))
        online_share.append(float(np.clip(on_share, 0.15, 0.85)))

    n_stages = len(buckets)
    protection = [0.0] * n_stages
    if n_stages > 0:
        protection[0] = float(np.clip((mus[0] + 1.2 * sigmas[0]) / max(config.initial_inventory, 1), 0.0, 0.98))
    cum_mu = 0.0
    cum_var = 0.0
    for j in range(1, n_stages):
        cum_mu += mus[j - 1]
        cum_var += sigmas[j - 1] ** 2
        fare_hi = max(fares[j - 1], 1e-6)
        fare_lo = max(fares[j], 1e-6)
        critical_fractile = float(np.clip(1.0 - fare_lo / fare_hi, 0.01, 0.99))
        z = _norm_ppf(critical_fractile)
        y = cum_mu + z * sqrt(max(cum_var, 1e-6))
        protection[j] = float(np.clip(y / max(config.initial_inventory, 1), 0.0, 0.98))

    stage_price_factor: List[float] = []
    for sid in range(n_stages):
        # 直接用历史 stage-ADR 相对全局 ADR 的比值作为价格因子
        # 比硬编码系数更有理论依据：需求强的阶段自然定高价
        fac = fares[sid] / max(global_adr, 1e-6)
        stage_price_factor.append(float(np.clip(fac, 0.80, 1.35)))

    return EMSRbProfile(
        global_adr=global_adr,
        season_weekday_factor=season_weekday_factor,
        stage_price_factor=stage_price_factor,
        stage_online_share=online_share,
    )


def _build_stage_policy_fn(config: Experiment2Config, profile: EMSRbProfile):
    def stage_policy_fn(stage_id: int, st: dict):
        sid = int(np.clip(stage_id, 0, len(profile.stage_price_factor) - 1))
        season = int(np.clip(int(st.get("season", 1)), 0, 2))
        weekday = int(np.clip(int(st.get("weekday", 0)), 0, 1))

        base = (
            profile.global_adr
            * profile.season_weekday_factor.get((season, weekday), 1.0)
            * profile.stage_price_factor[sid]
        )
        # 线上价格 = 基础价 × (1 - OTA佣金率)，线下无佣金定原价
        pon = base * (1.0 - float(config.commission_rate))
        poff = base

        pon = float(np.clip(pon, config.online_price_min, config.online_price_max))
        poff = float(np.clip(poff, config.offline_price_min, config.offline_price_max))
        return pon, poff

    return stage_policy_fn


def build_emsrb_init_mean_table(
    historical_data: pd.DataFrame,
    *,
    initial_inventory: int,
    booking_window_days: int,
    decision_buckets: str,
    online_price_min: float,
    online_price_max: float,
    offline_price_min: float,
    offline_price_max: float,
) -> Dict[Tuple[int, int, int], Tuple[float, float]]:
    """
    基于 EMSR-b 构造 CEM 的 coarse 初始化均值表。

    返回的键为 `(stage_id, season, weekday)`，用于给同一粗分组下的细状态共享初始均值。
    """
    config = Experiment2Config(
        initial_inventory=int(initial_inventory),
        booking_window_days=int(booking_window_days),
        decision_buckets=str(decision_buckets),
        online_price_min=float(online_price_min),
        online_price_max=float(online_price_max),
        offline_price_min=float(offline_price_min),
        offline_price_max=float(offline_price_max),
    )
    profile = _build_emsrb_profile(config, historical_data)
    stage_policy_fn = _build_stage_policy_fn(config, profile)
    n_stages = len(parse_buckets(config.decision_buckets, config.booking_window_days))

    table: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    for stage_id in range(n_stages):
        for season in (0, 1, 2):
            for weekday in (0, 1):
                pon, poff = stage_policy_fn(stage_id, {"season": season, "weekday": weekday})
                table[(int(stage_id), int(season), int(weekday))] = (float(pon), float(poff))
    return table


def _run_single_seed(
    config: Experiment2Config,
    historical_data: pd.DataFrame,
    seed: int,
) -> tuple[List[Dict], List[Dict], int]:
    train_records: List[Dict] = []
    eval_records: List[Dict] = []

    profile = _build_emsrb_profile(config, historical_data)
    stage_policy_fn = _build_stage_policy_fn(config, profile)
    eval_rewards = evaluate_policy(
        config=config,
        historical_data=historical_data,
        seed=seed + 500_000,
        stage_policy_fn=stage_policy_fn,
        n_episodes=config.post_eval_episodes,
    )
    for idx, rew in enumerate(eval_rewards, start=1):
        eval_records.append(
            {
                "Algorithm": "EMSR-b Heuristic",
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalHotelRevenue": float(rew["EvalHotelRevenue"]),
                "EvalOTAProfit": float(rew["EvalOTAProfit"]),
                "EvalSystemProfit": float(rew["EvalSystemProfit"]),
                "EvalRevenue": float(rew["EvalHotelRevenue"]),
            }
        )
    return train_records, eval_records, seed


def run_emsrb(config: Experiment2Config, historical_data: pd.DataFrame) -> tuple[List[Dict], List[Dict]]:
    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []
    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="EMSR-b Seeds", unit="seed"):
            _, seed_eval_records, _seed = _run_single_seed(config, historical_data, seed)
            all_eval_records.extend(seed_eval_records)
            tqdm.write(f"[EMSR-b] Seed {_seed} done: eval_ep={len(seed_eval_records)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed, config, historical_data, seed))

        with tqdm(total=len(futures), desc="EMSR-b Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                _, seed_eval_records, seed = fut.result()
                all_eval_records.extend(seed_eval_records)
                pbar.update(1)
                tqdm.write(f"[EMSR-b] Seed {seed} done: eval_ep={len(seed_eval_records)}")
    return all_train_records, all_eval_records
