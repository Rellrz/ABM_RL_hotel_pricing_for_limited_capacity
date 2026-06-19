"""
Cross-Entropy Method (CEM) 算法

基于采样的优化方法，通过迭代更新动作分布来找到最优策略。
适合连续动作空间，非常稳定。

核心思想：
1. 维护每个状态的动作分布 N(μ(s), σ(s))
2. 采样多个动作并评估
3. 选择top-k个最好的动作
4. 更新分布参数

优势：
- 不需要梯度
- 非常稳定
- 适合随机环境
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from configs.config import RL_CONFIG
from src.algorithms.base_algorithm import BaseRLAlgorithm


class CrossEntropyMethod(BaseRLAlgorithm):
    """
    Cross-Entropy Method (CEM) 算法
    
    特点：
    1. 基于采样的优化
    2. 维护高斯分布参数
    3. 通过精英样本更新分布
    4. 支持连续动作空间
    """
    
    def __init__(self,
                 n_states: int,
                 action_min: float = 80.0,
                 action_max: float = 170.0,
                 discount_factor: float = 0.99,
                 n_samples: int = 20,
                 elite_frac: float = 0.2,
                 initial_std: float = 20.0,
                 min_std: float = 2.0,
                 std_decay: float = 0.99,
                 memory_size: int = 100,
                 alpha: float | None = None):
        """
        初始化CEM算法
        
        Args:
            n_states: 状态空间大小
            action_min: 最小动作值
            action_max: 最大动作值
            discount_factor: 折扣因子
            n_samples: 每次采样的动作数量
            elite_frac: 精英样本比例（top-k）
            initial_std: 初始标准差
            min_std: 最小标准差
            std_decay: 标准差衰减率
            memory_size: 经验回放大小
            alpha: 分布参数平滑更新率；为空时从RL_CONFIG.cem_alpha读取
        """
        self.n_states = n_states
        self.action_min = action_min
        self.action_max = action_max
        self.discount_factor = discount_factor
        self.n_samples = n_samples
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.initial_std = initial_std
        self.min_std = min_std
        self.std_decay = std_decay
        self.memory_size = memory_size
        self.alpha = float(np.clip(RL_CONFIG.cem_alpha if alpha is None else alpha, 0.0, 1.0))
        
        # 动作分布参数：μ(s) 和 σ(s)
        initial_mean = (action_min + action_max) / 2.0
        self.mean_table = defaultdict(lambda: initial_mean)
        self.std_table = defaultdict(lambda: initial_std)
        
        # 经验回放：存储(state, action, reward)
        self.memory = defaultdict(lambda: deque(maxlen=memory_size))
        
        # 统计信息
        self.state_visit_count = defaultdict(int)
        self.episode_count = 0
        self.update_count = 0
    
    def select_action(self, state: Union[List, np.ndarray, int], 
                     deterministic: bool = False) -> float:
        """
        选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            连续动作值（价格）
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        mean = self.mean_table[state_key]
        std = self.std_table[state_key]
        
        if deterministic:
            # 确定性策略：返回均值
            action = mean
        else:
            # 随机策略：从高斯分布采样
            action = np.random.normal(mean, std)
        
        # 裁剪到有效范围
        action = np.clip(action, self.action_min, self.action_max)
        
        return float(action)
    
    def update(self, state: Union[List, np.ndarray, int], action: float,
              reward: float, next_state: Union[List, np.ndarray, int], 
              done: bool) -> float:
        """
        存储经验到回放缓冲区
        
        CEM不是每步更新，而是收集经验后批量更新
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            
        Returns:
            当前奖励（用于兼容接口）
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        # 存储经验
        self.memory[state_key].append({
            'action': action,
            'reward': reward,
            'done': done
        })
        
        # 更新访问计数
        self.state_visit_count[state_key] += 1
        
        return float(reward)
    
    def _update_distribution(self, state_key: Any):
        """
        更新状态的动作分布（CEM核心）
        
        1. 从经验中获取样本
        2. 选择top-k精英样本
        3. 更新均值和标准差
        
        Args:
            state_key: 状态键
        """
        if len(self.memory[state_key]) < self.n_elite:
            return  # 样本不足，不更新
        
        # 获取最近的经验
        recent_experiences = list(self.memory[state_key])[-self.n_samples:]
        
        if len(recent_experiences) < self.n_elite:
            return
        
        # 提取动作和奖励
        actions = np.array([exp['action'] for exp in recent_experiences])
        rewards = np.array([exp['reward'] for exp in recent_experiences])
        
        # 选择精英样本（奖励最高的top-k）
        elite_indices = np.argsort(rewards)[-self.n_elite:]
        elite_actions = actions[elite_indices]
        
        # 更新均值和标准差
        new_mean = np.mean(elite_actions)
        new_std = np.std(elite_actions)
        
        # 平滑更新（避免剧烈变化）
        alpha = self.alpha
        self.mean_table[state_key] = (1 - alpha) * self.mean_table[state_key] + alpha * new_mean
        self.std_table[state_key] = max(self.min_std, (1 - alpha) * self.std_table[state_key] + alpha * new_std)
        
        self.update_count += 1
    
    def end_episode(self):
        """
        结束一个episode
        
        在episode结束时更新所有访问过的状态的分布
        """
        self.episode_count += 1
        
        # 更新所有有足够经验的状态
        for state_key in self.memory.keys():
            self._update_distribution(state_key)
        
        # 衰减标准差
        for state_key in self.std_table.keys():
            self.std_table[state_key] = max(
                self.min_std,
                self.std_table[state_key] * self.std_decay
            )
    
    def get_policy(self) -> Dict[Any, float]:
        """
        获取当前策略（确定性）
        
        Returns:
            状态到最优动作（均值）的映射
        """
        policy = {}
        for state_key, mean in self.mean_table.items():
            policy[state_key] = mean
        return policy
    
    def get_value_stats(self) -> Dict[str, float]:
        """
        获取价值函数统计信息
        
        Returns:
            统计信息字典
        """
        # 计算均值统计
        means = list(self.mean_table.values())
        stds = list(self.std_table.values())
        
        # 估计Q值（使用平均奖励）
        avg_rewards = []
        for state_key, experiences in self.memory.items():
            if experiences:
                rewards = [exp['reward'] for exp in experiences]
                avg_rewards.append(np.mean(rewards))
        
        explored_states = len(self.mean_table)
        exploration_coverage = (explored_states / self.n_states * 100) if self.n_states > 0 else 0
        
        return {
            'num_states': explored_states,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'exploration_coverage': float(exploration_coverage),
            'mean_action_avg': float(np.mean(means)) if means else 0.0,
            'mean_action_std': float(np.std(means)) if means else 0.0,
            'mean_action_min': float(np.min(means)) if means else 0.0,
            'mean_action_max': float(np.max(means)) if means else 0.0,
            'std_avg': float(np.mean(stds)) if stds else 0.0,
            'std_min': float(np.min(stds)) if stds else 0.0,
            'std_max': float(np.max(stds)) if stds else 0.0,
            'avg_reward': float(np.mean(avg_rewards)) if avg_rewards else 0.0,
            'num_state_visits': sum(self.state_visit_count.values()),
            'mean_q_value': float(np.mean(avg_rewards)) if avg_rewards else 0.0,
            'std_q_value': float(np.std(avg_rewards)) if avg_rewards else 0.0,
            'min_q_value': float(np.min(avg_rewards)) if avg_rewards else 0.0,
            'max_q_value': float(np.max(avg_rewards)) if avg_rewards else 0.0,
            'zero_q_percentage': 0.0,
            'explored_state_actions': explored_states,
            'total_state_actions': self.n_states
        }
    
    def get_q_values(self, state: Union[List, np.ndarray, int]) -> float:
        """
        获取状态的Q值估计（用于兼容接口）
        
        Args:
            state: 状态
            
        Returns:
            Q值估计（使用平均奖励）
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        if state_key in self.memory and self.memory[state_key]:
            rewards = [exp['reward'] for exp in self.memory[state_key]]
            return float(np.mean(rewards))
        else:
            return 0.0
    
    def reset(self):
        """重置算法状态"""
        initial_mean = (self.action_min + self.action_max) / 2.0
        self.mean_table = defaultdict(lambda: initial_mean)
        self.std_table = defaultdict(lambda: self.initial_std)
        self.memory = defaultdict(lambda: deque(maxlen=self.memory_size))
        self.state_visit_count = defaultdict(int)
        self.episode_count = 0
        self.update_count = 0

    def save_model(self, file_name: str):
        """保存模型参数"""
        from configs.config import PATH_CONFIG
        from datetime import datetime
        import json, os

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        models_dir = PATH_CONFIG.models_dir

        save_path = os.path.join(models_dir, f'{file_name}_agent_{timestamp}.json')

        save_dict =  {'cem_online_means': {str(k): float(v) for k, v in self.mean_table.items()},
            'cem_online_stds': {str(k): float(v) for k, v in self.std_table.items()},
            'cem_online_state_visit_count': {str(k): int(v) for k, v in self.state_visit_count.items()}}
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        
