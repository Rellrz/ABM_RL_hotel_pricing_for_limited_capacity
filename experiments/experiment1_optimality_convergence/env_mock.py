"""实验一的极简 Mock 酒店环境。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from config import ExperimentConfig, DEFAULT_CONFIG


State = Tuple[int, int]


@dataclass
class StepResult:
    """便于在 MDP 估计与交互执行中复用的单步结果结构。"""

    next_state: State
    reward: float
    done: bool
    info: Dict[str, float]


class MockHotelEnv:
    """小规模、平稳的双渠道定价环境。

    该环境严格保持需求参数不随状态变化，从而对齐实验一的理论上界构建前提：
    对每个离散状态和离散动作，都可以通过高频 Monte Carlo 单步采样稳定估计
    转移概率与期望奖励，再交给动态规划求解。
    """

    def __init__(self, config: ExperimentConfig = DEFAULT_CONFIG, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state: State = (self.config.max_inv, self.config.max_lt)

    def seed(self, seed: Optional[int]) -> None:
        """重置环境内部随机数生成器。"""
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        state: Optional[State] = None,
    ) -> Tuple[State, Dict]:
        """重置环境，接口风格与 Gymnasium 保持一致。"""
        if seed is not None:
            self.seed(seed)
        self.state = state if state is not None else (self.config.max_inv, self.config.max_lt)
        return self.state, {}

    def set_state(self, state: State) -> None:
        """直接设置状态，便于 MDP 求解器遍历状态空间。"""
        inv, lt = state
        self.state = (int(inv), int(lt))

    def is_terminal(self, state: State) -> bool:
        """定义终止状态：库存为 0 或 lead_time 为 0。"""
        inv, lt = state
        return int(inv) == 0 or int(lt) == 0

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(action, dtype=float), self.config.price_min, self.config.price_max)

    def _demand_rate(self, price: float) -> float:
        """平稳线性需求：需求强度只由价格决定。"""
        return max(0.0, self.config.base_arrival_rate - self.config.price_sensitivity * float(price))

    def _allocate_sales(self, inventory: int, demand_on: int, demand_off: int) -> Tuple[int, int, int]:
        """按照需求比例分配有限库存。

        这里使用确定性的 floor 分配，避免在库存截断之后额外引入一层随机性，
        从而让 MDP 转移矩阵估计更加稳定。
        """
        total_demand = int(demand_on + demand_off)
        total_sold = min(int(inventory), total_demand)

        if total_sold <= 0 or total_demand <= 0:
            return 0, 0, 0

        sold_on = int(np.floor(total_sold * demand_on / total_demand))
        sold_on = max(0, min(sold_on, min(total_sold, demand_on)))
        sold_off = total_sold - sold_on

        # 双保险：保证分配结果不超过各渠道原始需求，若超出则回调到可行域。
        if sold_off > demand_off:
            overflow = sold_off - demand_off
            sold_off = demand_off
            sold_on = min(demand_on, sold_on + overflow)

        return sold_on, sold_off, total_sold

    def simulate_transition(
        self,
        state: State,
        action: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> StepResult:
        """从给定状态出发执行一步转移。

        该方法不依赖环境当前内部状态，适合 MDP 求解器和 rollout 逻辑复用。
        """
        local_rng = rng if rng is not None else self.rng
        inv, lt = int(state[0]), int(state[1])

        if self.is_terminal((inv, lt)):
            return StepResult(next_state=(inv, lt), reward=0.0, done=True, info={"demand_on": 0.0, "demand_off": 0.0})

        clipped_action = self._clip_action(action)
        p_online, p_offline = float(clipped_action[0]), float(clipped_action[1])

        demand_on = int(local_rng.poisson(self._demand_rate(p_online)))
        demand_off = int(local_rng.poisson(self._demand_rate(p_offline)))
        sold_on, sold_off, total_sold = self._allocate_sales(inv, demand_on, demand_off)

        reward = sold_on * p_online * (1.0 - self.config.commission_rate) + sold_off * p_offline
        next_inv = max(0, inv - total_sold)
        next_lt = max(0, lt - 1)
        next_state = (next_inv, next_lt)
        done = self.is_terminal(next_state)

        return StepResult(
            next_state=next_state,
            reward=float(reward),
            done=done,
            info={
                "demand_on": float(demand_on),
                "demand_off": float(demand_off),
                "sold_on": float(sold_on),
                "sold_off": float(sold_off),
                "price_online": p_online,
                "price_offline": p_offline,
            },
        )

    def step(self, action: np.ndarray) -> Tuple[State, float, bool, bool, Dict]:
        """执行一步环境交互。

        返回值对齐 Gymnasium 风格：
        `(obs, reward, terminated, truncated, info)`。
        """
        result = self.simulate_transition(self.state, action, rng=self.rng)
        self.state = result.next_state
        terminated = result.done
        truncated = False
        return self.state, result.reward, terminated, truncated, result.info
