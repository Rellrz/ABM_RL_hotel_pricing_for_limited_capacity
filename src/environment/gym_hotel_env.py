from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from configs.config import ENV_CONFIG
from src.environment.hotel_env import HotelEnvironment


class GymHotelPricingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
        capacity: Optional[int] = None,
        variable_cost_per_room: Optional[float] = None,
        reward_scale: Optional[float] = None,
        observation_mode: Optional[str] = None,
        scarcity_threshold_ratio: Optional[float] = None,
        scarcity_penalty_coef: Optional[float] = None,
        scarcity_penalty_weights: Optional[tuple[float, float, float] | list[float]] = None,
    ):
        super().__init__()
        self.base_seed = seed
        self.capacity = capacity
        self.variable_cost_per_room = variable_cost_per_room
        self.reward_scale = reward_scale
        self.observation_mode = observation_mode
        self.scarcity_threshold_ratio = scarcity_threshold_ratio
        self.scarcity_penalty_coef = scarcity_penalty_coef
        self.scarcity_penalty_weights = scarcity_penalty_weights
        self.env = HotelEnvironment(
            historical_data=historical_data,
            random_seed=seed,
            capacity=capacity,
            variable_cost_per_room=variable_cost_per_room,
            reward_scale=reward_scale,
            observation_mode=observation_mode,
            scarcity_threshold_ratio=scarcity_threshold_ratio,
            scarcity_penalty_coef=scarcity_penalty_coef,
            scarcity_penalty_weights=scarcity_penalty_weights,
        )
        self.price_min = float(ENV_CONFIG.price_min)
        self.price_max = float(ENV_CONFIG.price_max)
        self.action_space = spaces.Box(
            low=np.full(3, -1.0, dtype=np.float32),
            high=np.full(3, 1.0, dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        del options
        if seed is not None and seed != self.base_seed:
            self.env = HotelEnvironment(
                historical_data=self.env.abm_model.historical_data,
                random_seed=seed,
                capacity=self.capacity,
                variable_cost_per_room=self.variable_cost_per_room,
                reward_scale=self.reward_scale,
                observation_mode=self.observation_mode,
                scarcity_threshold_ratio=self.scarcity_threshold_ratio,
                scarcity_penalty_coef=self.scarcity_penalty_coef,
                scarcity_penalty_weights=self.scarcity_penalty_weights,
            )
            self.base_seed = seed
        self.env.reset()
        return self.env.get_state_vector(), {}

    def _denormalize_action(self, action) -> np.ndarray:
        scaled_action = np.asarray(action, dtype=np.float32).reshape(3)
        scaled_action = np.clip(scaled_action, -1.0, 1.0)
        midpoint = 0.5 * (self.price_min + self.price_max)
        half_range = 0.5 * (self.price_max - self.price_min)
        return midpoint + half_range * scaled_action

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        mapped_action = self._denormalize_action(action)
        _, reward, done, info = self.env.step(mapped_action)
        info["normalized_action"] = np.asarray(action, dtype=np.float32).reshape(3).tolist()
        info["mapped_price_action"] = mapped_action.astype(float).tolist()
        return self.env.get_state_vector(), float(reward), bool(done), False, info

    def render(self):
        return None
