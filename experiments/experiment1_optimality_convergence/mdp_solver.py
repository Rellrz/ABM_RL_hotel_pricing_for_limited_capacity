"""实验一的经验 MDP 求解器。"""

from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from config import ExperimentConfig, DEFAULT_CONFIG
from env_mock import MockHotelEnv, State


def _estimate_single_state_action(
    config: ExperimentConfig,
    action: np.ndarray,
    state: State,
    sample_seeds: np.ndarray,
) -> Tuple[int, int, np.ndarray, float]:
    """独立估计单个 `(s, a)` 的经验转移与奖励。

    该函数定义在模块级，便于被 joblib 的进程后端序列化。
    """
    env = MockHotelEnv(config)
    n_inv = config.n_inventory_states
    n_lt = config.n_lead_time_states
    inv, lt = int(state[0]), int(state[1])
    counts = np.zeros((n_inv, n_lt), dtype=np.int32)
    reward_sum = 0.0

    for sample_seed in sample_seeds:
        sample_rng = np.random.default_rng(int(sample_seed))
        result = env.simulate_transition((inv, lt), action, rng=sample_rng)
        next_inv, next_lt = result.next_state
        counts[next_inv, next_lt] += 1
        reward_sum += result.reward

    probs = counts / float(config.n_mc_samples)
    avg_reward = reward_sum / float(config.n_mc_samples)
    return inv, lt, probs, float(avg_reward)


class MDPSolver:
    """基于 Monte Carlo 经验估计构建小规模离散动作 MDP，并用值迭代求解。"""

    def __init__(self, config: ExperimentConfig = DEFAULT_CONFIG):
        self.config = config
        self.action_grid = self._build_action_grid()
        self.num_actions = len(self.action_grid)

    def _build_action_grid(self) -> np.ndarray:
        """生成 10 个整数价格点，并形成 100 组动作组合。"""
        points = np.rint(
            np.linspace(self.config.price_min, self.config.price_max, self.config.grid_size)
        ).astype(int)

        actions = []
        for p_online in points:
            for p_offline in points:
                actions.append([float(p_online), float(p_offline)])
        return np.asarray(actions, dtype=float)

    def _state_shape(self) -> Tuple[int, int]:
        return self.config.n_inventory_states, self.config.n_lead_time_states

    def _estimate_batch(
        self,
        batch_specs: List[Tuple[int, int, int, np.ndarray]],
    ) -> List[Tuple[int, int, int, np.ndarray, float]]:
        parallel = Parallel(n_jobs=self.config.n_jobs, prefer="processes")
        batch_results = parallel(
            delayed(_estimate_single_state_action)(
                self.config,
                self.action_grid[action_idx],
                (inv, lt),
                sample_seeds,
            )
            for inv, lt, action_idx, sample_seeds in batch_specs
        )

        merged = []
        for (inv, lt, action_idx, _), (_, _, probs, avg_reward) in zip(batch_specs, batch_results):
            merged.append((inv, lt, action_idx, probs, avg_reward))
        return merged

    def estimate_transition_and_reward(self, force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """若存在缓存则加载，否则重新估计经验转移矩阵与奖励矩阵。"""
        self.config.ensure_directories()

        if self.config.cache_path.exists() and not force_recompute:
            with open(self.config.cache_path, "rb") as f:
                payload = pickle.load(f)

            metadata = payload.get("metadata", {})
            expected = {
                "max_inv": self.config.max_inv,
                "max_lt": self.config.max_lt,
                "grid_size": self.config.grid_size,
                "n_mc_samples": self.config.n_mc_samples,
                "commission_rate": self.config.commission_rate,
                "base_arrival_rate": self.config.base_arrival_rate,
                "price_sensitivity": self.config.price_sensitivity,
                "stationary_env": True,
            }
            if metadata == expected:
                return payload["P"], payload["R"]

        n_inv, n_lt = self._state_shape()
        P = np.zeros((n_inv, n_lt, self.num_actions, n_inv, n_lt), dtype=np.float32)
        R = np.zeros((n_inv, n_lt, self.num_actions), dtype=np.float32)

        env = MockHotelEnv(self.config)
        non_terminal_pairs = [
            (inv, lt, a_idx)
            for inv in range(n_inv)
            for lt in range(n_lt)
            for a_idx in range(self.num_actions)
            if not env.is_terminal((inv, lt))
        ]

        for inv in range(n_inv):
            for lt in range(n_lt):
                if env.is_terminal((inv, lt)):
                    P[inv, lt, :, inv, lt] = 1.0
                    R[inv, lt, :] = 0.0

        seed_rng = np.random.default_rng(self.config.base_seed)
        batch_size = max(self.config.n_jobs * 4, 16)
        current_batch: List[Tuple[int, int, int, np.ndarray]] = []

        for inv, lt, action_idx in tqdm(
            non_terminal_pairs,
            desc="Estimating MDP P/R",
            unit="(s,a)",
        ):
            sample_seeds = seed_rng.integers(
                0,
                2**32 - 1,
                size=self.config.n_mc_samples,
                dtype=np.uint32,
            )
            current_batch.append((inv, lt, action_idx, sample_seeds))

            if len(current_batch) >= batch_size:
                for b_inv, b_lt, b_action_idx, probs, avg_reward in self._estimate_batch(current_batch):
                    P[b_inv, b_lt, b_action_idx] = probs
                    R[b_inv, b_lt, b_action_idx] = avg_reward
                current_batch = []

        if current_batch:
            for b_inv, b_lt, b_action_idx, probs, avg_reward in self._estimate_batch(current_batch):
                P[b_inv, b_lt, b_action_idx] = probs
                R[b_inv, b_lt, b_action_idx] = avg_reward

        payload = {
            "P": P,
            "R": R,
            "action_grid": self.action_grid,
            "metadata": {
                "max_inv": self.config.max_inv,
                "max_lt": self.config.max_lt,
                "grid_size": self.config.grid_size,
                "n_mc_samples": self.config.n_mc_samples,
                "commission_rate": self.config.commission_rate,
                "base_arrival_rate": self.config.base_arrival_rate,
                "price_sensitivity": self.config.price_sensitivity,
                "stationary_env": True,
            },
        }
        with open(self.config.cache_path, "wb") as f:
            pickle.dump(payload, f)

        return P, R

    def value_iteration(self, P: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """在离散动作 MDP 上执行值迭代。"""
        n_inv, n_lt = self._state_shape()
        V = np.zeros((n_inv, n_lt), dtype=np.float64)
        policy_idx = np.zeros((n_inv, n_lt), dtype=np.int32)

        env = MockHotelEnv(self.config)
        for _ in range(self.config.vi_max_iter):
            V_new = V.copy()
            max_delta = 0.0

            for inv in range(n_inv):
                for lt in range(n_lt):
                    state: State = (inv, lt)
                    if env.is_terminal(state):
                        V_new[inv, lt] = 0.0
                        policy_idx[inv, lt] = 0
                        continue

                    q_values = np.empty(self.num_actions, dtype=np.float64)
                    for a_idx in range(self.num_actions):
                        q_values[a_idx] = R[inv, lt, a_idx] + self.config.gamma * np.sum(
                            P[inv, lt, a_idx] * V
                        )

                    best_idx = int(np.argmax(q_values))
                    V_new[inv, lt] = q_values[best_idx]
                    policy_idx[inv, lt] = best_idx

                    delta = abs(V_new[inv, lt] - V[inv, lt])
                    if delta > max_delta:
                        max_delta = delta

            V = V_new
            if max_delta < self.config.vi_tol:
                break

        return V, policy_idx

    def solve(self, force_recompute: bool = False) -> Dict:
        """一站式执行 MDP 预计算与值迭代。"""
        P, R = self.estimate_transition_and_reward(force_recompute=force_recompute)
        V, policy_idx = self.value_iteration(P, R)
        upper_bound = float(V[self.config.max_inv, self.config.max_lt])
        return {
            "P": P,
            "R": R,
            "V": V,
            "policy_idx": policy_idx,
            "upper_bound": upper_bound,
            "action_grid": self.action_grid,
        }
