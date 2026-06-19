"""
强化学习算法基类

定义所有RL算法的通用接口，确保算法之间的兼容性。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np


class BaseRLAlgorithm(ABC):
    """
    强化学习算法基类
    
    所有RL算法都应该继承这个基类并实现以下方法：
    - select_action: 选择动作
    - update: 更新算法参数
    - get_policy: 获取当前策略
    - get_value_stats: 获取价值函数统计信息
    """
    
    @abstractmethod
    def select_action(self, state: Union[List, np.ndarray, int], 
                     deterministic: bool = False) -> Union[int, float]:
        """
        选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略（测试时使用）
            
        Returns:
            动作（离散动作返回int，连续动作返回float）
        """
        pass
    
    @abstractmethod
    def update(self, state: Union[List, np.ndarray, int], action: Union[int, float],
              reward: float, next_state: Union[List, np.ndarray, int], 
              done: bool) -> float:
        """
        更新算法参数
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            
        Returns:
            更新后的目标值（如TD目标）
        """
        pass
    
    def end_episode(self):
        """
        结束一个episode
        
        用于更新episode级别的参数（如探索率衰减）
        """
        pass
    
    @abstractmethod
    def get_policy(self) -> Dict[Any, Union[int, float]]:
        """
        获取当前策略（确定性）
        
        Returns:
            状态到最优动作的映射
        """
        pass
    
    @abstractmethod
    def get_value_stats(self) -> Dict[str, float]:
        """
        获取价值函数统计信息
        
        Returns:
            统计信息字典，包含：
            - mean_q_value: 平均Q值
            - std_q_value: Q值标准差
            - min_q_value: 最小Q值
            - max_q_value: 最大Q值
            - exploration_coverage: 探索覆盖率
            等
        """
        pass
    
    def get_q_values(self, state: Union[List, np.ndarray, int]) -> Union[np.ndarray, float]:
        """
        获取状态的Q值（用于兼容接口）
        
        Args:
            state: 状态
            
        Returns:
            Q值数组（离散动作）或单个Q值（连续动作）
        """
        # 默认实现：返回0
        return 0.0
    
    def reset(self):
        """
        重置算法状态
        
        用于重新开始训练
        """
        pass
