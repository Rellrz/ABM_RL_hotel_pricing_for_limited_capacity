"""Independent CEM：去协方差消融版。"""

from __future__ import annotations

import numpy as np
from typing import Any

from configs.experiment2 import Experiment2Config
from src.algorithms.multivariate_cem import MultivariateCrossEntropyMethod


class IndependentCEMAgent:
    """基于 MCEM 的对角协方差版本（仅保留标准差，不建模相关性）。"""

    def __init__(self, config: Experiment2Config):
        self.config = config
        self.agent = MultivariateCrossEntropyMethod(
            n_states=config.q_n_states,
            action_mins=(config.online_price_min, config.offline_price_min),
            action_maxs=(config.online_price_max, config.offline_price_max),
            discount_factor=0.99,
            n_samples=config.cem_n_samples,
            elite_frac=config.cem_elite_frac,
            initial_std=config.cem_initial_std,
            min_std=config.cem_min_std,
            std_decay=config.cem_std_decay,
            memory_size=config.cem_memory_size,
            diagonal_covariance=True,
        )

    def select_action(self, state_idx: Any, deterministic: bool = False) -> np.ndarray:
        return self.agent.select_action(state_idx, deterministic=deterministic).astype(np.float64)

    def update(self, s: Any, a_pair: np.ndarray, r: float, s_next: Any, done: bool) -> None:
        self.agent.update(s, a_pair, float(r), s_next, done)

    def end_episode(self) -> None:
        self.agent.end_episode()
