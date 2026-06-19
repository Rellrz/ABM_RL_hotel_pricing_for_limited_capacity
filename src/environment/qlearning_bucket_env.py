"""Q-learning 用离散环境封装（状态144，动作10x10）。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from configs.experiment2 import Experiment2Config
from src.environment.bucket_pricing_simulator import BucketPricingSimulator


class QLearningBucketEnv:
    def __init__(self, config: Experiment2Config, seed: int, historical_data):
        self.config = config
        self.sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
        self.action_grid = config.q_action_grid
        self.n_actions = self.action_grid.shape[0]
        self.n_stages = self.sim.n_stages
        self.current_stage = 0

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            _ = int(seed)
        self.sim.reset()
        self.current_stage = 0
        return self.sim.get_q_state_by_stage(self.current_stage)

    def step(self, action_idx: int) -> Tuple[int, float, bool, dict]:
        action_idx = int(np.clip(action_idx, 0, self.n_actions - 1))
        stage_action = tuple(self.action_grid[action_idx].tolist())
        stage_actions = [stage_action for _ in range(self.n_stages)]
        result = self.sim.step_day(stage_actions)

        self.current_stage = (self.current_stage + 1) % self.n_stages
        next_state = self.sim.get_q_state_by_stage(self.current_stage)
        reward = float(result.reward_hotel)
        done = bool(result.done)
        return next_state, reward, done, result.info
