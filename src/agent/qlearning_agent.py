"""表格 Q-learning（状态144，动作100）。"""

from __future__ import annotations

import numpy as np

from src.utils.common import q_epsilon
from configs.experiment2 import Experiment2Config


class QLearningAgent:
    def __init__(self, config: Experiment2Config, seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((config.q_n_states, config.q_action_grid.shape[0]), dtype=np.float64)
        self.total_steps = 0

    def select_action(self, state_idx: int, deterministic: bool = False) -> int:
        if deterministic:
            return int(np.argmax(self.q[state_idx]))
        eps = q_epsilon(
            step=self.total_steps,
            eps_start=self.config.q_eps_start,
            eps_end=self.config.q_eps_end,
            decay_steps=self.config.q_eps_decay_steps,
        )
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.q.shape[1]))
        return int(np.argmax(self.q[state_idx]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        target = float(r)
        if not done:
            target += self.config.q_gamma * float(np.max(self.q[s_next]))
        self.q[s, a] = (1.0 - self.config.q_alpha) * self.q[s, a] + self.config.q_alpha * target
        self.total_steps += 1
