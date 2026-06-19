"""实验一所需的三类智能体。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from config import ExperimentConfig, DEFAULT_CONFIG
from env_mock import MockHotelEnv, State


Trajectory = Dict[str, object]


class BaseAgent:
    """统一约定智能体接口。"""

    def act(self, state: State) -> np.ndarray:
        del state
        raise NotImplementedError

    def select_action(self, state: State, deterministic: bool = False) -> np.ndarray:
        del deterministic
        return self.act(state)

    def update(self, trajectories: List[Trajectory]) -> None:
        del trajectories
        return None


class RandomAgent(BaseAgent):
    """随机连续动作基线。"""

    def __init__(self, config: ExperimentConfig = DEFAULT_CONFIG, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def act(self, state: State) -> np.ndarray:
        del state
        return self.rng.uniform(self.config.price_min, self.config.price_max, size=2)


class StaticAgent(BaseAgent):
    """固定动作基线。

    先用统一搜索种子在动作网格上做一次静态搜索，
    后续所有 seeds 共用同一组固定价格。
    """

    def __init__(self, action: np.ndarray):
        self.fixed_action = np.asarray(action, dtype=float)

    def act(self, state: State) -> np.ndarray:
        del state
        return self.fixed_action.copy()

    @classmethod
    def from_grid_search(
        cls,
        config: ExperimentConfig,
        action_grid: np.ndarray,
    ) -> "StaticAgent":
        env = MockHotelEnv(config)
        rng = np.random.default_rng(config.static_search_seed)

        best_action = action_grid[0]
        best_value = -np.inf

        for action in tqdm(action_grid, desc="Searching static baseline", unit="action"):
            rollout_rewards = []
            for _ in range(config.static_eval_rollouts):
                rollout_seed = int(rng.integers(0, 2**32 - 1))
                total_reward = rollout_episode(env, cls(action), rollout_seed)
                rollout_rewards.append(total_reward)

            mean_reward = float(np.mean(rollout_rewards))
            if mean_reward > best_value:
                best_value = mean_reward
                best_action = action.copy()

        return cls(best_action)


class CEMAgent(BaseAgent):
    """更贴近主项目 mCEM 的状态条件二维高斯智能体。

    与前一版 trajectory-level CEM 的差别：
    1. 为每个状态维护独立的经验缓冲区；
    2. 将轨迹拆解为状态级样本，并使用 reward-to-go 做信用分配；
    3. 训练时采样动作，评估时使用分布均值动作。
    """

    def __init__(self, config: ExperimentConfig = DEFAULT_CONFIG, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

        n_inv = self.config.n_inventory_states
        n_lt = self.config.n_lead_time_states

        self.mu = np.zeros((n_inv, n_lt, 2), dtype=np.float64)
        self.cov = np.zeros((n_inv, n_lt, 2, 2), dtype=np.float64)
        self.memory = defaultdict(lambda: deque(maxlen=self.config.memory_size))

        for inv in range(n_inv):
            for lt in range(n_lt):
                self.mu[inv, lt] = self.config.init_mean
                self.cov[inv, lt] = self.config.init_cov

        self.last_update_summary: Dict[str, float] = {}

    def _sanitize_cov(self, cov: np.ndarray) -> np.ndarray:
        """将协方差矩阵投影回对称正定邻域。"""
        cov = np.asarray(cov, dtype=np.float64)
        cov = (cov + cov.T) / 2.0

        min_var = self.config.min_std ** 2
        cov[0, 0] = max(cov[0, 0], min_var)
        cov[1, 1] = max(cov[1, 1], min_var)
        cov = cov + np.eye(2) * self.config.cov_reg

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_var + self.config.cov_reg)
        return (eigvecs * eigvals) @ eigvecs.T

    def select_action(self, state: State, deterministic: bool = False) -> np.ndarray:
        inv, lt = int(state[0]), int(state[1])
        if deterministic:
            action = self.mu[inv, lt]
        else:
            action = self.rng.multivariate_normal(
                self.mu[inv, lt],
                self._sanitize_cov(self.cov[inv, lt]),
            )
        return np.clip(action, self.config.price_min, self.config.price_max)

    def act(self, state: State) -> np.ndarray:
        return self.select_action(state, deterministic=False)

    def _append_trajectory_to_memory(self, trajectory: Trajectory) -> None:
        transitions = trajectory["transitions"]
        reward_to_go = 0.0

        for transition in reversed(transitions):
            reward_to_go = float(transition["reward"]) + self.config.gamma * reward_to_go
            state = transition["state"]
            self.memory[state].append(
                {
                    "action": np.asarray(transition["action"], dtype=np.float64),
                    "score": reward_to_go,
                }
            )

    def _update_distribution(self, state: State) -> bool:
        if len(self.memory[state]) < self.config.min_update_samples:
            return False

        recent = list(self.memory[state])[-self.config.memory_size :]
        if not recent:
            return False

        actions = np.asarray([item["action"] for item in recent], dtype=np.float64)
        scores = np.asarray([item["score"] for item in recent], dtype=np.float64)
        n_elite = max(1, int(np.ceil(len(recent) * self.config.elite_frac)))
        elite_idx = np.argsort(scores)[-n_elite:]
        elite_actions = actions[elite_idx]

        mu_hat = np.mean(elite_actions, axis=0)
        if len(elite_actions) >= 2:
            cov_hat = np.cov(elite_actions, rowvar=False)
        else:
            cov_hat = np.diag([self.config.min_std ** 2, self.config.min_std ** 2]).astype(np.float64)

        cov_hat = self._sanitize_cov(cov_hat)
        inv, lt = int(state[0]), int(state[1])
        self.mu[inv, lt] = np.clip(
            (1.0 - self.config.alpha) * self.mu[inv, lt] + self.config.alpha * mu_hat,
            self.config.price_min,
            self.config.price_max,
        )
        self.cov[inv, lt] = self._sanitize_cov(
            ((1.0 - self.config.alpha) * self.cov[inv, lt] + self.config.alpha * cov_hat)
            * (self.config.std_decay ** 2)
        )
        self._smooth_to_neighbors((inv, lt))
        return True

    def _smooth_to_neighbors(self, state: State) -> None:
        """向相邻状态做轻微平滑，提升样本效率并保持策略表连续。"""
        beta = self.config.neighbor_smoothing
        if beta <= 0.0:
            return

        inv, lt = int(state[0]), int(state[1])
        source_mu = self.mu[inv, lt].copy()
        source_cov = self.cov[inv, lt].copy()

        neighbors = [
            (inv - 1, lt),
            (inv + 1, lt),
            (inv, lt - 1),
            (inv, lt + 1),
        ]
        for n_inv, n_lt in neighbors:
            if not (0 <= n_inv < self.config.n_inventory_states and 0 <= n_lt < self.config.n_lead_time_states):
                continue

            self.mu[n_inv, n_lt] = np.clip(
                (1.0 - beta) * self.mu[n_inv, n_lt] + beta * source_mu,
                self.config.price_min,
                self.config.price_max,
            )
            self.cov[n_inv, n_lt] = self._sanitize_cov(
                (1.0 - beta) * self.cov[n_inv, n_lt] + beta * source_cov
            )

    def update(self, trajectories: List[Trajectory]) -> None:
        """先把轨迹写入状态缓冲区，再执行状态级 elite 更新。"""
        if not trajectories:
            return

        n_elite = max(1, int(np.ceil(len(trajectories) * self.config.elite_frac)))
        elite_batch = sorted(trajectories, key=lambda x: float(x["total_reward"]), reverse=True)[:n_elite]

        for trajectory in elite_batch:
            self._append_trajectory_to_memory(trajectory)

        updated_states = 0
        for state in list(self.memory.keys()):
            if self._update_distribution(state):
                updated_states += 1

        self.last_update_summary = {
            "elite_count": float(n_elite),
            "updated_states": float(updated_states),
            "mean_trajectory_reward": float(np.mean([t["total_reward"] for t in trajectories])),
            "elite_mean_reward": float(np.mean([t["total_reward"] for t in elite_batch])),
        }


def rollout_episode(
    env: MockHotelEnv,
    agent: BaseAgent,
    episode_seed: int,
    deterministic: bool = False,
) -> float:
    """执行一条完整轨迹，仅返回累计回报。"""
    state, _ = env.reset(seed=episode_seed)
    total_reward = 0.0

    while True:
        action = agent.select_action(state, deterministic=deterministic)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break

    return float(total_reward)


def rollout_episode_with_trace(
    env: MockHotelEnv,
    agent: BaseAgent,
    episode_seed: int,
    deterministic: bool = False,
) -> Trajectory:
    """执行一条完整轨迹，并保留状态-动作轨迹用于 CEM 更新。"""
    state, _ = env.reset(seed=episode_seed)
    total_reward = 0.0
    transitions = []

    while True:
        action = agent.select_action(state, deterministic=deterministic)
        next_state, reward, terminated, truncated, _ = env.step(action)
        transitions.append(
            {
                "state": state,
                "action": np.asarray(action, dtype=np.float64),
                "reward": float(reward),
                "next_state": next_state,
                "done": bool(terminated or truncated),
            }
        )
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break

    return {"transitions": transitions, "total_reward": float(total_reward)}
