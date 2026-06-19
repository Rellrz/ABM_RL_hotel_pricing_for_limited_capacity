"""Genetic Algorithm 基线：基于 pymoo 的种群进化搜索最优固定价格表。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

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


class _HotelPricingProblem(Problem):
    """pymoo Problem：最小化 -HotelRevenue。"""

    def __init__(
        self,
        config: Experiment2Config,
        historical_data,
        search_seed: int,
        eval_side_channel: list,
        pbar: tqdm,
    ):
        xl = np.array([float(config.online_price_min)] * 8 + [float(config.offline_price_min)] * 8)
        xu = np.array([float(config.online_price_max)] * 8 + [float(config.offline_price_max)] * 8)
        super().__init__(n_var=16, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self._config = config
        self._historical_data = historical_data
        self._search_seed = search_seed
        self._eval_side_channel = eval_side_channel
        self._pbar = pbar

    def _evaluate(self, x, out, *args, **kwargs):
        # pymoo 以向量化方式传入 X，shape: (pop_size, 16)
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        hotels, otas, systems = [], [], []
        for i in range(x.shape[0]):
            theta = x[i]
            hotel, ota, system = _evaluate_theta(
                theta, self._config, self._historical_data, self._search_seed, n_episodes=1,
            )
            hotels.append(-hotel)
            otas.append(ota)
            systems.append(system)
            self._eval_side_channel.append((hotel, ota, system))
            self._pbar.update(1)
            self._pbar.set_postfix({"best": f"{-min(hotels):.0f}"})
        out["F"] = np.array(hotels).reshape(-1, 1)


def _run_single_seed_ga(
    config: Experiment2Config,
    historical_data,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """单种子 GA 搜索 + 最终评估。"""
    train_records: List[Dict] = []
    eval_records: List[Dict] = []

    search_seed = seed + 10_000
    pop_size = int(getattr(config, "ga_pop_size", 40))
    n_gen = int(getattr(config, "ga_n_generations", 50))

    # Side-channel 收集每代每个个体的完整评估结果
    eval_log: list[tuple[float, float, float]] = []

    total_evals = pop_size * n_gen
    pbar = tqdm(
        total=total_evals,
        desc=f"GA Seed {seed}",
        unit="eval",
        leave=False,
    )

    problem = _HotelPricingProblem(
        config=config,
        historical_data=historical_data,
        search_seed=search_seed,
        eval_side_channel=eval_log,
        pbar=pbar,
    )

    algorithm = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(
            prob=float(getattr(config, "ga_crossover_prob", 0.9)),
            eta=float(getattr(config, "ga_crossover_eta", 15.0)),
        ),
        mutation=PM(
            eta=float(getattr(config, "ga_mutation_eta", 20.0)),
        ),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    pbar.close()

    # 训练记录：按评估顺序展平
    for i, (hotel, ota, system) in enumerate(eval_log, start=1):
        train_records.append(
            {
                "Algorithm": "GA",
                "Seed": seed,
                "Episode": i,
                "EpisodeHotelRevenue": hotel,
                "EpisodeOTAProfit": ota,
                "EpisodeSystemProfit": system,
                "EpisodeRevenue": hotel,
            }
        )

    # 最终评估：最优 theta × post_eval_episodes
    best_theta = np.asarray(result.X, dtype=np.float64)
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
                "Algorithm": "GA",
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalHotelRevenue": float(rew["EvalHotelRevenue"]),
                "EvalOTAProfit": float(rew["EvalOTAProfit"]),
                "EvalSystemProfit": float(rew["EvalSystemProfit"]),
                "EvalRevenue": float(rew["EvalHotelRevenue"]),
            }
        )

    return train_records, eval_records


def run_ga(
    config: Experiment2Config,
    historical_data,
) -> Tuple[List[Dict], List[Dict]]:
    """GA 基线主入口，支持多 seed 并行。"""
    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []

    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="GA Seeds", unit="seed"):
            train_rec, eval_rec = _run_single_seed_ga(config, historical_data, seed)
            all_train_records.extend(train_rec)
            all_eval_records.extend(eval_rec)
            tqdm.write(f"[GA] Seed {seed} done: train_iters={len(train_rec)} eval_ep={len(eval_rec)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed_ga, config, historical_data, seed))

        with tqdm(total=len(futures), desc="GA Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                train_rec, eval_rec = fut.result()
                all_train_records.extend(train_rec)
                all_eval_records.extend(eval_rec)
                pbar.update(1)

    return all_train_records, all_eval_records
