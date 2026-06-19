"""
Multivariate Cross-Entropy Method (CEM) for joint hotel pricing.

This implementation models a 2D action:
    action = [price_online_base, price_offline]
with a per-state multivariate Gaussian distribution:
    N(mu_s, Sigma_s)
"""

from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np


class MultivariateCrossEntropyMethod:
    """
    State-conditional multivariate CEM with Gaussian sampling.

    Notes:
    - Each state maintains a 2D mean vector and 2x2 covariance matrix.
    - Covariance is sanitized after each update to ensure PSD stability.
    """

    def __init__(
        self,
        n_states: int,
        action_mins: Tuple[float, float],
        action_maxs: Tuple[float, float],
        discount_factor: float = 0.99,
        n_samples: int = 100,
        elite_frac: float = 0.3,
        initial_std: Union[float, Tuple[float, float]] = 20.0,
        min_std: float = 2.0,
        std_decay: float = 0.99,
        memory_size: int = 100,
        alpha: float = 0.3,
        cov_reg: float = 1e-5,
        diagonal_covariance: bool = False,
        initial_mean_provider: Callable[[Any], np.ndarray | List[float] | Tuple[float, float] | None] | None = None,
    ):
        self.n_states = int(n_states)
        self.action_mins = np.array(action_mins, dtype=float)
        self.action_maxs = np.array(action_maxs, dtype=float)
        self.discount_factor = float(discount_factor)
        self.n_samples = int(n_samples)
        self.n_elite = max(1, int(self.n_samples * float(elite_frac)))
        self.min_std = float(min_std)
        self.std_decay = float(std_decay)
        self.memory_size = int(memory_size)
        self.alpha = float(alpha)
        self.cov_reg = float(cov_reg)
        self.diagonal_covariance = bool(diagonal_covariance)
        self.initial_mean_provider = initial_mean_provider

        if np.isscalar(initial_std):
            init_std_vec = np.array([float(initial_std), float(initial_std)], dtype=float)
        else:
            init_std_vec = np.array(initial_std, dtype=float)
            if init_std_vec.shape != (2,):
                raise ValueError(f"initial_std must be scalar or shape (2,), got {init_std_vec.shape}")

        self.initial_std_vec = np.maximum(init_std_vec, self.min_std)
        self.initial_cov = np.diag(self.initial_std_vec ** 2)
        self.initial_mean = (self.action_mins + self.action_maxs) / 2.0

        self.mean_table: Dict[Any, np.ndarray] = {}
        self.cov_table: Dict[Any, np.ndarray] = {}
        self.memory = defaultdict(lambda: deque(maxlen=self.memory_size))
        self.state_visit_count = defaultdict(int)
        self.episode_count = 0
        self.update_count = 0

    def _state_key(self, state: Union[List, np.ndarray, int]) -> Any:
        return tuple(state) if isinstance(state, (list, np.ndarray)) else state

    def _initial_mean_for_state(self, state_key: Any) -> np.ndarray:
        if self.initial_mean_provider is None:
            return self.initial_mean.copy()
        candidate = self.initial_mean_provider(state_key)
        if candidate is None:
            return self.initial_mean.copy()
        arr = np.asarray(candidate, dtype=float).reshape(-1)
        if arr.shape != (2,):
            return self.initial_mean.copy()
        return np.clip(arr, self.action_mins, self.action_maxs).astype(float)

    def _ensure_state_params(self, state_key: Any) -> None:
        if state_key not in self.mean_table:
            self.mean_table[state_key] = self._initial_mean_for_state(state_key)
        if state_key not in self.cov_table:
            self.cov_table[state_key] = self.initial_cov.copy()

    def _sanitize_cov(self, cov: np.ndarray) -> np.ndarray:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (2, 2):
            cov = self.initial_cov.copy()

        # Enforce symmetry.
        cov = (cov + cov.T) / 2.0

        # Ensure minimum diagonal variance and regularization.
        min_var = self.min_std ** 2
        d = np.diag(cov).copy()
        d = np.maximum(d, min_var)
        cov[0, 0], cov[1, 1] = d[0], d[1]
        cov[0, 0] += self.cov_reg
        cov[1, 1] += self.cov_reg

        if self.diagonal_covariance:
            cov[0, 1] = 0.0
            cov[1, 0] = 0.0
            return cov

        # Project to PSD by clipping eigenvalues.
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, self.cov_reg)
        cov = (vecs * vals) @ vecs.T
        cov = (cov + cov.T) / 2.0

        # Re-apply diagonal floor after projection.
        d = np.diag(cov).copy()
        d = np.maximum(d, min_var + self.cov_reg)
        cov[0, 0], cov[1, 1] = d[0], d[1]
        return cov

    def sample(self, state: Union[List, np.ndarray, int], n_samples: int = None) -> np.ndarray:
        state_key = self._state_key(state)
        self._ensure_state_params(state_key)
        mu = self.mean_table[state_key]
        cov = self._sanitize_cov(self.cov_table[state_key])
        num = self.n_samples if n_samples is None else int(n_samples)

        try:
            actions = np.random.multivariate_normal(mean=mu, cov=cov, size=num)
        except np.linalg.LinAlgError:
            cov = self._sanitize_cov(cov + np.eye(2) * (10 * self.cov_reg))
            actions = np.random.multivariate_normal(mean=mu, cov=cov, size=num)

        return np.clip(actions, self.action_mins, self.action_maxs)

    def select_action(self, state: Union[List, np.ndarray, int], deterministic: bool = False) -> np.ndarray:
        state_key = self._state_key(state)
        self._ensure_state_params(state_key)
        if deterministic:
            action = self.mean_table[state_key]
        else:
            action = self.sample(state_key, n_samples=1)[0]
        return np.clip(action, self.action_mins, self.action_maxs).astype(float)

    def update(
        self,
        state: Union[List, np.ndarray, int],
        action: Union[List[float], np.ndarray],
        reward: float,
        next_state: Union[List, np.ndarray, int],
        done: bool,
    ) -> float:
        state_key = self._state_key(state)
        self._ensure_state_params(state_key)
        action_arr = np.asarray(action, dtype=float).reshape(-1)
        if action_arr.shape != (2,):
            raise ValueError(f"action must be shape (2,), got {action_arr.shape}")

        self.memory[state_key].append(
            {
                "action": np.clip(action_arr, self.action_mins, self.action_maxs),
                "reward": float(reward),
                "done": bool(done),
            }
        )
        self.state_visit_count[state_key] += 1
        return float(reward)

    def _update_distribution(self, state_key: Any) -> bool:
        recent = list(self.memory[state_key])[-self.n_samples :]
        if len(recent) < self.n_elite:
            return False

        actions = np.array([exp["action"] for exp in recent], dtype=float)  # (N,2)
        rewards = np.array([exp["reward"] for exp in recent], dtype=float)  # (N,)

        elite_idx = np.argsort(rewards)[-self.n_elite :]
        elites = actions[elite_idx]  # (K,2)

        mu_hat = np.mean(elites, axis=0)
        cov_hat = np.cov(elites, rowvar=False)
        if np.ndim(cov_hat) == 0:
            cov_hat = np.diag([float(cov_hat), float(cov_hat)])
        elif cov_hat.shape == (2,):
            cov_hat = np.diag(cov_hat)
        if self.diagonal_covariance:
            cov_hat = np.diag(np.diag(cov_hat))
        cov_hat = self._sanitize_cov(cov_hat)

        mu_old = self.mean_table[state_key]
        cov_old = self.cov_table[state_key]

        mu_new = (1.0 - self.alpha) * mu_old + self.alpha * mu_hat
        cov_new = (1.0 - self.alpha) * cov_old + self.alpha * cov_hat
        cov_new = self._sanitize_cov(cov_new)

        self.mean_table[state_key] = np.clip(mu_new, self.action_mins, self.action_maxs)
        self.cov_table[state_key] = cov_new
        self.update_count += 1
        return True

    def end_episode(self) -> None:
        self.episode_count += 1
        updated_states = []
        for state_key in list(self.memory.keys()):
            if self._update_distribution(state_key):
                updated_states.append(state_key)

        # 仅对本轮实际完成参数更新的状态执行协方差衰减
        decay_scale = self.std_decay ** 2
        for state_key in updated_states:
            decayed = self.cov_table[state_key] * decay_scale
            self.cov_table[state_key] = self._sanitize_cov(decayed)

    def get_policy(self) -> Dict[Any, List[float]]:
        return {k: self.mean_table[k].tolist() for k in self.mean_table.keys()}

    def get_exploration_scale(self) -> float:
        if not self.cov_table:
            return 1.0
        stds = []
        for cov in self.cov_table.values():
            c = self._sanitize_cov(cov)
            stds.append(np.sqrt(np.diag(c)))
        stds_arr = np.array(stds, dtype=float)
        avg_std = float(np.mean(stds_arr))
        init_std = float(np.mean(self.initial_std_vec))
        return avg_std / max(init_std, 1e-8)

    def save_model(self, file_name: str) -> str:
        from configs.config import PATH_CONFIG
        from datetime import datetime
        import json
        import os

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(PATH_CONFIG.models_dir, f"{file_name}_agent_{timestamp}.json")

        save_dict = {
            "algo": "multivariate_cem_diag" if self.diagonal_covariance else "multivariate_cem",
            "action_mins": self.action_mins.tolist(),
            "action_maxs": self.action_maxs.tolist(),
            "initial_std_vec": self.initial_std_vec.tolist(),
            "min_std": self.min_std,
            "std_decay": self.std_decay,
            "alpha": self.alpha,
            "cov_reg": self.cov_reg,
            "diagonal_covariance": self.diagonal_covariance,
            "memory_size": self.memory_size,
            "n_samples": self.n_samples,
            "n_elite": self.n_elite,
            "cem_state_fields": ["stage_id", "season", "weekday", "near_inv_bin", "far_inv_bin"],
            "means": {str(k): v.tolist() for k, v in self.mean_table.items()},
            "covs": {str(k): v.tolist() for k, v in self.cov_table.items()},
            "state_visit_count": {str(k): int(v) for k, v in self.state_visit_count.items()},
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)
        return save_path
