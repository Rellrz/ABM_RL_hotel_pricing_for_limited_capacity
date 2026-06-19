"""
随机因子配置模块

用于控制系统中的随机性，支持以下模式：
1. 固定模式：设置具体随机种子，确保结果可复现
2. 随机模式：不设置随机种子，允许随机变化

使用说明：
- 将random_mode设置为'fixed'或'random'
- 在fixed模式下，使用fixed_seed值作为随机种子
- 在random模式下，不设置任何随机种子
"""

import numpy as np
import random
import time

# 从config.py导入随机因子配置
from configs.config import RANDOM_CONFIG

def setup_random_factors():
    """
    根据配置设置随机因子
    
    Returns:
        dict: 当前随机因子配置信息
    """
    config = RANDOM_CONFIG
    
    if config.random_mode == 'fixed':
        # 固定模式：设置随机种子
        np.random.seed(config.fixed_seed)
        random.seed(config.fixed_seed)
        config.current_status = f"固定模式 - 种子: {config.fixed_seed}"
        print(f"随机因子已设置为固定模式，种子: {config.fixed_seed}")
    else:
        # 随机模式：使用时间戳作为种子，确保每次运行都不同
        time_seed = int(time.time() * 1000) % (2**32)  # 使用毫秒级时间戳
        np.random.seed(time_seed)
        random.seed(time_seed)
        config.current_status = f"随机模式 - 时间戳种子: {time_seed}"
        print(f"随机因子已设置为随机模式，使用时间戳种子: {time_seed}")
    
    return config

# 自动设置随机因子（模块导入时执行）
current_random_config = setup_random_factors()

if __name__ == "__main__":
    # 测试随机因子配置
    print("=== 随机因子配置测试 ===")
    print(f"当前模式: {current_random_config['random_mode']}")
    print(f"状态: {current_random_config['current_status']}")
    
    # 生成随机数测试
    print("\n随机数测试:")
    print(f"numpy随机数: {np.random.rand(3)}")
    print(f"python随机数: {[random.random() for _ in range(3)]}")