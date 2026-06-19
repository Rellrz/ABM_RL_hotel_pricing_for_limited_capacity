"""Random Search 基线：纯随机采样最优固定价格表，定义零智能搜索下界。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from configs.experiment2 import Experiment2Config
from src.evaluation.policy_evaluator import StagePolicyFn, evaluate_policy


def _theta_to_stage_policy_fn(theta: np.ndarray) -> StagePolicyFn:
    """将 16 维价格向量转换为 stage_policy_fn。"""
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
    """评估单个价格向量 theta，返回 (hotel, ota, system)。"""
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


def _run_single_seed_rs(
    config: Experiment2Config,
    historical_data,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """单种子 Random Search + 最终评估。"""
    train_records: List[Dict] = []
    eval_records: List[Dict] = []

    search_seed = seed + 10_000
    n_iter = int(getattr(config, "rs_n_iterations", 2000))

    low = np.array([float(config.online_price_min)] * 8 + [float(config.offline_price_min)] * 8)
    high = np.array([float(config.online_price_max)] * 8 + [float(config.offline_price_max)] * 8)

    best_theta = None
    best_hotel = -float("inf")

    pbar = tqdm(
        total=n_iter,
        desc=f"RS Seed {seed}",
        unit="sample",
        leave=False,
    )

    rng = np.random.RandomState(search_seed)
    for i in range(1, n_iter + 1):
        theta = rng.uniform(low, high)
        hotel, ota, system = _evaluate_theta(
            theta, config, historical_data, search_seed, n_episodes=1,
        )

        train_records.append(
            {
                "Algorithm": "RS",
                "Seed": seed,
                "Episode": i,
                "EpisodeHotelRevenue": hotel,
                "EpisodeOTAProfit": ota,
                "EpisodeSystemProfit": system,
                "EpisodeRevenue": hotel,
            }
        )

        if hotel > best_hotel:
            best_hotel = hotel
            best_theta = theta.copy()

        pbar.update(1)
        pbar.set_postfix({"best": f"{best_hotel:.0f}"})

    pbar.close()

    # 最终评估：最优 theta × post_eval_episodes
    if best_theta is None:
        best_theta = (low + high) / 2.0  # fallback

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
                "Algorithm": "RS",
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalHotelRevenue": float(rew["EvalHotelRevenue"]),
                "EvalOTAProfit": float(rew["EvalOTAProfit"]),
                "EvalSystemProfit": float(rew["EvalSystemProfit"]),
                "EvalRevenue": float(rew["EvalHotelRevenue"]),
            }
        )

    return train_records, eval_records


def run_rs(
    config: Experiment2Config,
    historical_data,
) -> Tuple[List[Dict], List[Dict]]:
    """Random Search 基线主入口，支持多 seed 并行。"""
    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []

    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="RS Seeds", unit="seed"):
            train_rec, eval_rec = _run_single_seed_rs(config, historical_data, seed)
            all_train_records.extend(train_rec)
            all_eval_records.extend(eval_rec)
            tqdm.write(f"[RS] Seed {seed} done: train_iters={len(train_rec)} eval_ep={len(eval_rec)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed_rs, config, historical_data, seed))

        with tqdm(total=len(futures), desc="RS Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                train_rec, eval_rec = fut.result()
                all_train_records.extend(train_rec)
                all_eval_records.extend(eval_rec)
                pbar.update(1)

    return all_train_records, all_eval_records
