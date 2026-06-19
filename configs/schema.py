#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional

from .base import PROJECT_ROOT


@dataclass
class PathConfig:
    """路径配置"""
    raw_data_dir: str = os.path.join(PROJECT_ROOT, 'data', 'raw')
    processed_data_dir: str = os.path.join(PROJECT_ROOT, 'data', 'processed')
    hotel_bookings_csv: str = os.path.join(PROJECT_ROOT, 'data', 'raw', 'hotel_bookings.csv')

    models_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'models')
    results_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'results')
    figures_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
    tensorboard_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'tensorboard_logs')

    def __post_init__(self):
        for path in [self.raw_data_dir, self.processed_data_dir, self.models_dir, self.results_dir, self.figures_dir]:
            os.makedirs(path, exist_ok=True)


@dataclass
class ABMConfig:
    """ABM客户行为模型配置"""
    monthly_arrival_rates: Dict[int, float] = field(default_factory=lambda: {m: 100.0 for m in range(1, 13)})
    # 到达率按月份与日类型分层：0=工作日，1=节假日（当前用周末代理）
    arrival_rate_by_month_daytype: Dict[int, Dict[int, float]] = field(
        default_factory=lambda: {m: {0: 100.0, 1: 100.0} for m in range(1, 13)}
    )
    lead_time_params: Dict[str, Any] = field(default_factory=lambda: {'type': 'exponential', 'mean': 104.0})
    wtp_params: Dict[str, float] = field(default_factory=lambda: {'mean': 100.0, 'std': 30.0})
    room_marginal_cost: float = 10.0

    urgency_weight: float = 20
    noise_std: float = 12.0
    booking_threshold: float = -15
    customer_type_ratio: Tuple[float, float] = (0.7, 0.3)  # (online_only, omnichannel)
    online_discount_ratio: float = 0.95

    anchor_enabled: bool = True
    anchor_quantile_low: float = 0.10
    anchor_quantile_high: float = 0.90
    
    anchor_weight_low: float = 0.0
    anchor_weight_mean: float = 1
    anchor_weight_high: float = 0.0
    
    # anchor强度
    anchor_eta: float = 0.1
    anchor_joint_theta: float = 0.50
    anchor_lambda_plus: float = 1.0
    anchor_lambda_minus: float = 2.0

    regret_coefficient: float = 0.75
    commitment_weight: float = 8.0
    shock_std: float = 15.0

    beta_base: float = 1.0
    #价格敏感度
    beta_range: Tuple[float, float] = (0.8, 1.2)

    enable_perturbation: bool = False
    perturbation_seed: Optional[int] = None

    demand_ou_theta: float = 0.15
    demand_ou_sigma: float = 0.08
    demand_jump_prob: float = 0.02
    demand_jump_mean: float = 0.20
    demand_jump_std: float = 0.10
    lambda_multiplier_min: float = 0.60
    lambda_multiplier_max: float = 1.80

    wtp_ou_theta: float = 0.20
    wtp_ou_sigma: float = 0.03
    wtp_multiplier_min: float = 0.80
    wtp_multiplier_max: float = 1.20
    wtp_std_multiplier_min: float = 0.90
    wtp_std_multiplier_max: float = 1.30

    channel_pref_ou_theta: float = 0.25
    channel_pref_ou_sigma: float = 0.04
    channel_online_only_prob_min: float = 0.10
    channel_online_only_prob_max: float = 0.90

    utility_noise_type: str = 'gumbel'
    utility_gumbel_beta: float = 0.8
    utility_normal_std: float = 1.0


@dataclass
class RLConfig:
    """强化学习配置（博弈主线，仅支持CEM/CEM-NN）"""
    n_states: int = 18
    initial_std: float = 50.0
    min_std: float = 3.0
    std_decay: float = 0.999

    reward_hotel_ratio: float = 1
    reward_ota_ratio: float = 0
    reward_shape_price_weight: float = 0.3            #0.3
    reward_shape_sellthrough_weight: float = 0.22      #0.22
    reward_shape_target_sellthrough: float = 0.25      #0.25

    cem_algorithm: str = 'cem'
    cem_n_samples: int = 400
    cem_elite_frac: float = 0.3
    cem_alpha: float = 0.2
    cem_init_strategy: str = 'midpoint'  # midpoint / emsrb / blended
    cem_init_blend_alpha: float = 0.7

    cem_nn_state_dim: int = 18
    cem_nn_learning_rate: float = 0.001
    cem_nn_batch_size: int = 32
    cem_nn_memory_size: int = 1000
    cem_nn_hidden_dims: list = field(default_factory=lambda: [64, 64])
    cem_nn_min_std: float = 0.02
    cem_nn_initial_std: float = 0.1

    enable_game_mode: bool = False
    commission_rate: float = 0.20
    subsidy_ratio_min: float = 0.0
    subsidy_ratio_max: float = 0.8
    ota_delta_max: float = 15.0
    ota_decay_lambda: float = 0.05
    ota_noise_std: float = 0.05
    ota_seed: int = 42
    online_price_min: float = 50.0
    online_price_max: float = 150.0
    offline_price_min: float = 50.0
    offline_price_max: float = 150.0
    game_training_mode: str = 'simultaneous'

    episodes: int = 250
    online_learning_days: int = 90
    update_frequency: int = 90
    
    cem_memory_size: int = 400
    enable_online_learning: bool = False


@dataclass
class EnvConfig:
    """酒店环境参数"""
    initial_inventory: int = 150
    booking_window_days: int = 91


@dataclass
class SimulationConfig:
    """系统模拟和评估参数"""
    default_days: int = 100
    default_start_date: str = '2017-01-01'
    evaluation_episodes: int = 10
    results_path: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, '04_结果输出', 'simulation_results'))


@dataclass
class RandomConfig:
    """控制系统中的随机性"""
    random_mode: str = 'random'
    fixed_seed: int = 42
    description: str = '随机因子控制配置 - 支持固定和随机两种模式'


@dataclass
class SystemConfig:
    """系统级配置参数"""
    use_cuda: bool = False
    device: str = 'cpu'
    random_seed: int = 42
    max_workers: int = 28
    memory_limit_gb: int = 24
    enable_gpu_memory_growth: bool = True
    mixed_precision: bool = False
    compile_models: bool = False
    profile_performance: bool = False


@dataclass
class LogConfig:
    """系统日志和输出配置"""
    log_level: str = 'INFO'
    log_file: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, '06_临时文件', 'hotel_pricing.log'))
    console_output: bool = True
    save_intermediate_results: bool = True
