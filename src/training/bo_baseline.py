"""Bayesian Optimization 基线：使用高斯过程 + Expected Improvement 搜索最优固定价格表。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from tqdm import tqdm

from configs.experiment2 import Experiment2Config
from src.evaluation.policy_evaluator import StagePolicyFn, evaluate_policy


def _theta_to_stage_policy_fn(theta: np.ndarray) -> StagePolicyFn:
    """将 16 维价格向量转换为 stage_policy_fn。

    theta[:8]  = 线上基础价 (bucket 0..7)
    theta[8:]  = 线下价 (bucket 0..7)
    """
    theta = np.asarray(theta, dtype=np.float64)

    def stage_policy_fn(stage_id: int, _st: dict) -> Tuple[float, float]:
        sid = int(np.clip(stage_id, 0, 7))
        return float(theta[sid]), float(theta[sid + 8])

    return stage_policy_fn


def _evaluate_theta(
    theta: np.ndarray,
    config: Experiment2Config,
    historical_data,
    base_seed: int,
    n_episodes: int = 1,
) -> tuple[float, float, float]:
    """评估单个价格向量 theta。

    返回 (平均酒店收入, 平均OTA利润, 平均系统利润)，供 BO 和训练记录使用。
    """
    stage_policy_fn = _theta_to_stage_policy_fn(theta)
    results = evaluate_policy(
        config=config,
        historical_data=historical_data,
        seed=base_seed,
        stage_policy_fn=stage_policy_fn,
        n_episodes=n_episodes,
    )
    hotel = float(np.mean([float(r["EvalHotelRevenue"]) for r in results]))
    ota = float(np.mean([float(r["EvalOTAProfit"]) for r in results]))
    system = float(np.mean([float(r["EvalSystemProfit"]) for r in results]))
    return hotel, ota, system


def _run_single_seed_bo(
    config: Experiment2Config,
    historical_data,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """单种子 BO 搜索 + 最终评估。"""
    train_records: List[Dict] = []
    eval_records: List[Dict] = []

    # 定义 16 维搜索空间
    dimensions = [
        Real(float(config.online_price_min), float(config.online_price_max), name=f"online_{i}")
        for i in range(8)
    ] + [
        Real(float(config.offline_price_min), float(config.offline_price_max), name=f"offline_{i}")
        for i in range(8)
    ]

    # BO 搜索阶段使用独立 seed 偏移，避免与最终评估 seed 重叠
    search_seed = seed + 10_000
    n_ep_per_point = int(getattr(config, "bo_n_eval_episodes_per_point", 1))

    # 用列表收集每轮评估的详细结果（side-channel），避免重复评估
    eval_results_side: list[tuple[float, float, float]] = []

    n_calls = int(getattr(config, "bo_n_calls", 200))
    n_initial = int(min(getattr(config, "bo_n_initial_points", 20), n_calls))

    pbar = tqdm(
        total=n_calls,
        desc=f"BO Seed {seed}",
        unit="eval",
        leave=False,
    )

    def objective(theta: np.ndarray) -> float:
        hotel, ota, system = _evaluate_theta(
            theta, config, historical_data, search_seed, n_episodes=n_ep_per_point,
        )
        eval_results_side.append((hotel, ota, system))
        pbar.update(1)
        pbar.set_postfix({"revenue": f"{hotel:.0f}"})
        return -hotel  # gp_minimize 最小化酒店收入

    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        acq_func=str(getattr(config, "bo_acq_func", "EI")),
        noise="gaussian",
        random_state=seed,
    )
    pbar.close()

    # 记录训练过程（BO 迭代序列）
    for i, (hotel, ota, system) in enumerate(eval_results_side, start=1):
        train_records.append(
            {
                "Algorithm": "BO",
                "Seed": seed,
                "Episode": i,
                "EpisodeHotelRevenue": hotel,
                "EpisodeOTAProfit": ota,
                "EpisodeSystemProfit": system,
                "EpisodeRevenue": hotel,
            }
        )

    # 最终评估：最优 theta × post_eval_episodes
    best_theta = np.asarray(result.x, dtype=np.float64)
    stage_policy_fn = _theta_to_stage_policy_fn(best_theta)
    final_eval = evaluate_policy(
        config=config,
        historical_data=historical_data,
        seed=seed + 500_000,
        stage_policy_fn=stage_policy_fn,
        n_episodes=config.post_eval_episodes,
    )
    for idx, rew in enumerate(final_eval, start=1):
        eval_records.append(
            {
                "Algorithm": "BO",
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalHotelRevenue": float(rew["EvalHotelRevenue"]),
                "EvalOTAProfit": float(rew["EvalOTAProfit"]),
                "EvalSystemProfit": float(rew["EvalSystemProfit"]),
                "EvalRevenue": float(rew["EvalHotelRevenue"]),
            }
        )

    return train_records, eval_records


def run_bo(
    config: Experiment2Config,
    historical_data,
) -> Tuple[List[Dict], List[Dict]]:
    """BO 基线主入口，支持多 seed 并行。

    接口与 run_emsrb / run_cem_family 一致。
    """
    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []

    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="BO Seeds", unit="seed"):
            train_rec, eval_rec = _run_single_seed_bo(config, historical_data, seed)
            all_train_records.extend(train_rec)
            all_eval_records.extend(eval_rec)
            tqdm.write(f"[BO] Seed {seed} done: train_iters={len(train_rec)} eval_ep={len(eval_rec)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed_bo, config, historical_data, seed))

        with tqdm(total=len(futures), desc="BO Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                train_rec, eval_rec = fut.result()
                all_train_records.extend(train_rec)
                all_eval_records.extend(eval_rec)
                pbar.update(1)

    return all_train_records, all_eval_records
