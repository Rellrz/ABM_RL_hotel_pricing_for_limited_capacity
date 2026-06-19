"""实验一的并行运行器。"""

from __future__ import annotations

from contextlib import contextmanager
import json
import multiprocessing as mp
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from agents import CEMAgent, RandomAgent, StaticAgent, rollout_episode, rollout_episode_with_trace
from config import ExperimentConfig, DEFAULT_CONFIG
from env_mock import MockHotelEnv
from mdp_solver import MDPSolver

try:
    from joblib import Parallel, delayed
    import joblib.parallel as joblib_parallel
    from joblib.parallel import BatchCompletionCallBack

    HAS_JOBLIB = True
except ImportError:
    Parallel = None
    delayed = None
    joblib_parallel = None
    BatchCompletionCallBack = None
    HAS_JOBLIB = False


def _seed_list(config: ExperimentConfig) -> List[int]:
    return [config.base_seed + i for i in range(config.n_seeds)]


def _dispatch_job(args: Tuple[str, int, ExperimentConfig, np.ndarray | None]) -> pd.DataFrame:
    algorithm, seed, config, static_action = args
    if algorithm == "CEM":
        return _run_cem_seed(seed, config)
    return _run_fixed_policy_seed(algorithm, seed, config, static_action=static_action)


@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """让 joblib 在批任务完成时正确更新 tqdm 进度条。

    直接把 tqdm 包在任务生成器外层，只能显示“任务已提交”，不能显示“任务已完成”。
    这里通过覆写 joblib 的批量完成回调，把进度更新绑定到真实完成事件上。
    """
    if not HAS_JOBLIB or BatchCompletionCallBack is None:
        yield tqdm_object
        return

    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib_parallel.BatchCompletionCallBack
    joblib_parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib_parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def _run_cem_seed(seed: int, config: ExperimentConfig) -> pd.DataFrame:
    env = MockHotelEnv(config)
    agent = CEMAgent(config=config, seed=seed)
    rng = np.random.default_rng(seed)
    records = []

    for episode in range(1, config.n_episodes + 1):
        trajectories = []
        rollout_rewards = []
        for _ in range(config.n_rollouts_per_episode):
            rollout_seed = int(rng.integers(0, 2**32 - 1))
            trajectory = rollout_episode_with_trace(env, agent, rollout_seed)
            trajectories.append(trajectory)

        agent.update(trajectories)
        if config.deterministic_eval:
            for _ in range(config.n_rollouts_per_episode):
                eval_seed = int(rng.integers(0, 2**32 - 1))
                rollout_rewards.append(
                    rollout_episode(
                        env,
                        agent,
                        eval_seed,
                        deterministic=True,
                    )
                )
        else:
            rollout_rewards = [float(trajectory["total_reward"]) for trajectory in trajectories]

        records.append(
            {
                "Algorithm": "CEM",
                "Seed": seed,
                "Episode": episode,
                "Total_Reward": float(np.mean(rollout_rewards)),
            }
        )

    return pd.DataFrame(records)


def _run_fixed_policy_seed(
    algorithm: str,
    seed: int,
    config: ExperimentConfig,
    static_action: np.ndarray | None = None,
) -> pd.DataFrame:
    env = MockHotelEnv(config)
    rng = np.random.default_rng(seed)
    records = []

    if algorithm == "Static":
        if static_action is None:
            raise ValueError("Static algorithm requires a precomputed fixed action.")
        agent = StaticAgent(static_action)
    elif algorithm == "Random":
        agent = RandomAgent(config=config, seed=seed)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    for episode in range(1, config.n_episodes + 1):
        rollout_rewards = []
        for _ in range(config.n_rollouts_per_episode):
            rollout_seed = int(rng.integers(0, 2**32 - 1))
            total_reward = rollout_episode(env, agent, rollout_seed)
            rollout_rewards.append(total_reward)

        records.append(
            {
                "Algorithm": algorithm,
                "Seed": seed,
                "Episode": episode,
                "Total_Reward": float(np.mean(rollout_rewards)),
            }
        )

    return pd.DataFrame(records)


class ExperimentRunner:
    """负责多随机种子并行执行三类算法。"""

    def __init__(self, config: ExperimentConfig = DEFAULT_CONFIG):
        self.config = config
        self.config.ensure_directories()

    def compute_static_action(self) -> np.ndarray:
        """全局搜索一次静态最优固定动作，供全部 seeds 共用。"""
        solver = MDPSolver(self.config)
        static_agent = StaticAgent.from_grid_search(self.config, solver.action_grid)
        return static_agent.fixed_action

    def _parallel_run(self, jobs: List[Tuple[str, int]], static_action: np.ndarray) -> pd.DataFrame:
        payloads = [(algorithm, seed, self.config, static_action) for algorithm, seed in jobs]

        if HAS_JOBLIB:
            progress_bar = tqdm(total=len(payloads), desc="Running experiment seeds", unit="job")
            with tqdm_joblib(progress_bar):
                results = Parallel(n_jobs=self.config.n_jobs, backend="loky")(
                    delayed(_dispatch_job)(payload) for payload in payloads
                )
        elif self.config.n_jobs > 1:
            with mp.Pool(processes=self.config.n_jobs) as pool:
                results = list(
                    tqdm(
                        pool.imap(_dispatch_job, payloads),
                        total=len(payloads),
                        desc="Running experiment seeds",
                        unit="job",
                    )
                )
        else:
            results = [
                _dispatch_job(payload)
                for payload in tqdm(payloads, desc="Running experiment seeds", unit="job")
            ]

        return pd.concat(results, ignore_index=True)

    def run_all(self) -> Tuple[pd.DataFrame, Dict]:
        """执行三类算法、保存 CSV，并返回汇总信息。"""
        static_action = self.compute_static_action()
        seeds = _seed_list(self.config)
        jobs = [(algorithm, seed) for algorithm in ["CEM", "Static", "Random"] for seed in seeds]

        df_results = self._parallel_run(jobs, static_action=static_action)
        df_results.to_csv(self.config.results_csv_path, index=False)

        summary = {
            "static_action": [float(static_action[0]), float(static_action[1])],
            "n_rows": int(len(df_results)),
            "algorithms": sorted(df_results["Algorithm"].unique().tolist()),
            "seeds": seeds,
            "cem_deterministic_eval": bool(self.config.deterministic_eval),
        }
        with open(self.config.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return df_results, summary
