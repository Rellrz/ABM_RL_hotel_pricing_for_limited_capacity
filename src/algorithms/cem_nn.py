"""
Neural Network Cross-Entropy Method (CEM-NN) 算法

使用神经网络来学习状态到动作的映射，相比表格版本CEM：
1. 可以处理高维连续状态空间
2. 具有更好的泛化能力
3. 适合大规模问题

核心思想：
1. 使用神经网络输出动作分布的均值和标准差 N(μ(s), σ(s))
2. 采样多个动作并评估
3. 选择top-k个最好的动作
4. 通过监督学习更新网络参数

优势：
- 不需要离散化状态空间
- 可以泛化到未见过的状态
- 适合连续状态和动作空间
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.base_algorithm import BaseRLAlgorithm


class PolicyNetwork(nn.Module):
    """
    策略网络：输出动作分布的均值和对数标准差
    
    输入：状态向量
    输出：(mean, log_std)
    """
    
    def __init__(self, state_dim: int, action_dim: int = 1, 
                 hidden_dims: List[int] = [64, 64],
                 action_min: float = 80.0,
                 action_max: float = 170.0):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（默认1，单一价格）
            hidden_dims: 隐藏层维度列表
            action_min: 最小动作值
            action_max: 最大动作值
        """
        super(PolicyNetwork, self).__init__()
        
        self.action_min = action_min
        self.action_max = action_max
        self.action_dim = action_dim
        
        # 构建网络层
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 均值头
        self.mean_head = nn.Linear(input_dim, action_dim)
        
        # 对数标准差头
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        x = self.shared_layers(state)
        
        # 均值：使用tanh激活并缩放到[action_min, action_max]
        mean = self.mean_head(x)
        mean = torch.tanh(mean)
        mean = self.action_min + (mean + 1.0) * 0.5 * (self.action_max - self.action_min)
        
        # 标准差：使用softplus确保为正，并限制范围
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-5, max=2)  # 限制log_std范围
        std = torch.exp(log_std)
        
        return mean, std


class NeuralCrossEntropyMethod(BaseRLAlgorithm):
    """
    Neural Network Cross-Entropy Method (CEM-NN) 算法
    
    特点：
    1. 使用神经网络学习策略
    2. 基于采样的优化
    3. 通过精英样本监督学习
    4. 支持连续状态和动作空间
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 1,
                 action_min: float = 80.0,
                 action_max: float = 170.0,
                 discount_factor: float = 0.99,
                 n_samples: int = 20,
                 elite_frac: float = 0.2,
                 learning_rate: float = 0.001,
                 hidden_dims: List[int] = [64, 64],
                 batch_size: int = 32,
                 memory_size: int = 1000,
                 min_std: float = 2.0,
                 initial_std: float = None,
                 device: str = 'cpu'):
        """
        初始化CEM-NN算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            action_min: 最小动作值
            action_max: 最大动作值
            discount_factor: 折扣因子
            n_samples: 每次采样的动作数量
            elite_frac: 精英样本比例（top-k）
            learning_rate: 学习率
            hidden_dims: 隐藏层维度列表
            batch_size: 批次大小
            memory_size: 经验回放大小
            min_std: 最小标准差
            initial_std: 初始标准差（如果为None，则使用min_std*5）
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.discount_factor = discount_factor
        self.n_samples = n_samples
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.min_std = min_std
        self.initial_std = initial_std if initial_std is not None else min_std * 5
        self.device = torch.device(device)
        
        # 初始化策略网络
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_min=action_min,
            action_max=action_max
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 经验回放：存储(state, action, reward)
        self.memory = deque(maxlen=memory_size)
        
        # 统计信息
        self.state_visit_count = {}
        self.episode_count = 0
        self.update_count = 0
        self.total_loss = 0.0
    
    def _state_to_tensor(self, state: Union[List, np.ndarray, int]) -> torch.Tensor:
        """
        将状态转换为张量
        
        Args:
            state: 状态（可以是int、list或ndarray）
            
        Returns:
            状态张量 [1, state_dim]
        """
        if isinstance(state, int):
            # 离散状态：转换为one-hot编码
            state_vec = np.zeros(self.state_dim)
            if state < self.state_dim:
                state_vec[state] = 1.0
            else:
                # 如果状态超出范围，使用最后一个维度
                state_vec[-1] = 1.0
        elif isinstance(state, (list, np.ndarray)):
            state_vec = np.array(state, dtype=np.float32)
            # 如果维度不匹配，进行填充或截断
            if len(state_vec) < self.state_dim:
                state_vec = np.pad(state_vec, (0, self.state_dim - len(state_vec)))
            elif len(state_vec) > self.state_dim:
                state_vec = state_vec[:self.state_dim]
        else:
            state_vec = np.zeros(self.state_dim)
        
        return torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
    
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
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            mean, std = self.policy_net(state_tensor)
            
            mean = mean.cpu().numpy()[0, 0]
            std = std.cpu().numpy()[0, 0]
            
            # 确保标准差不小于最小值
            std = max(std, self.min_std)
            
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
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            
        Returns:
            当前奖励（用于兼容接口）
        """
        # 存储经验
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done
        })
        
        # 更新访问计数
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        self.state_visit_count[state_key] = self.state_visit_count.get(state_key, 0) + 1
        
        return float(reward)
    
    def _update_policy(self):
        """
        更新策略网络（CEM核心）
        
        1. 从经验中获取样本
        2. 选择top-k精英样本
        3. 使用监督学习更新网络
        """
        if len(self.memory) < self.n_elite:
            return  # 样本不足，不更新
        
        # 获取最近的经验
        recent_experiences = list(self.memory)[-min(self.n_samples * 5, len(self.memory)):]
        
        if len(recent_experiences) < self.n_elite:
            return
        
        # 提取状态、动作和奖励
        states = [exp['state'] for exp in recent_experiences]
        actions = np.array([exp['action'] for exp in recent_experiences])
        rewards = np.array([exp['reward'] for exp in recent_experiences])
        
        # 选择精英样本（奖励最高的top-k）
        elite_indices = np.argsort(rewards)[-self.n_elite:]
        elite_states = [states[i] for i in elite_indices]
        elite_actions = actions[elite_indices]
        
        # 转换为张量
        state_tensors = torch.cat([self._state_to_tensor(s) for s in elite_states], dim=0)
        action_targets = torch.FloatTensor(elite_actions).unsqueeze(1).to(self.device)
        
        # 训练网络
        self.policy_net.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        mean_pred, std_pred = self.policy_net(state_tensors)
        
        # 损失函数：负对数似然（最大化精英动作的概率）
        # log p(a|s) = -0.5 * ((a - μ) / σ)^2 - log(σ) - 0.5 * log(2π)
        diff = (action_targets - mean_pred) / (std_pred + 1e-8)
        nll_loss = 0.5 * (diff ** 2) + torch.log(std_pred + 1e-8)
        loss = nll_loss.mean()
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        self.total_loss += loss.item()
    
    def end_episode(self):
        """
        结束一个episode
        
        在episode结束时更新策略网络
        """
        self.episode_count += 1
        
        # 更新策略网络
        self._update_policy()
    
    def get_policy(self) -> Dict[Any, float]:
        """
        获取当前策略（确定性）
        
        Returns:
            状态到最优动作的映射（仅返回访问过的状态）
        """
        policy = {}
        self.policy_net.eval()
        with torch.no_grad():
            for state_key in self.state_visit_count.keys():
                state_tensor = self._state_to_tensor(state_key)
                mean, _ = self.policy_net(state_tensor)
                policy[state_key] = float(mean.cpu().numpy()[0, 0])
        return policy
    
    def get_value_stats(self) -> Dict[str, float]:
        """
        获取价值函数统计信息
        
        Returns:
            统计信息字典
        """
        # 计算平均奖励
        avg_rewards = []
        if self.memory:
            rewards = [exp['reward'] for exp in self.memory]
            avg_rewards.append(np.mean(rewards))
        
        explored_states = len(self.state_visit_count)
        
        return {
            'num_states': explored_states,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'exploration_coverage': 0.0,  # NN版本不适用
            'avg_reward': float(np.mean(avg_rewards)) if avg_rewards else 0.0,
            'num_state_visits': sum(self.state_visit_count.values()),
            'mean_q_value': float(np.mean(avg_rewards)) if avg_rewards else 0.0,
            'std_q_value': float(np.std(avg_rewards)) if avg_rewards else 0.0,
            'min_q_value': float(np.min(avg_rewards)) if avg_rewards else 0.0,
            'max_q_value': float(np.max(avg_rewards)) if avg_rewards else 0.0,
            'zero_q_percentage': 0.0,
            'explored_state_actions': explored_states,
            'total_state_actions': 0,  # NN版本不适用
            'avg_loss': self.total_loss / max(1, self.update_count)
        }
    
    def get_q_values(self, state: Union[List, np.ndarray, int]) -> float:
        """
        获取状态的Q值估计（用于兼容接口）
        
        Args:
            state: 状态
            
        Returns:
            Q值估计（使用平均奖励）
        """
        if self.memory:
            rewards = [exp['reward'] for exp in self.memory if exp['state'] == state]
            if rewards:
                return float(np.mean(rewards))
        return 0.0
    
    def reset(self):
        """重置算法状态"""
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[64, 64],
            action_min=self.action_min,
            action_max=self.action_max
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=self.memory_size)
        self.state_visit_count = {}
        self.episode_count = 0
        self.update_count = 0
        self.total_loss = 0.0
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'update_count': self.update_count,
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']
        self.update_count = checkpoint['update_count']
