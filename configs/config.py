#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容入口：保留旧导入路径 `configs.config`，实际实现已拆分到子模块。"""

from .base import DATA_PATH, PROJECT_ROOT
from .defaults import ABM_PERTURBATION_TEMPLATES, apply_abm_perturbation_template
from .estimators import (
    build_empirical_lead_time_distribution,
    calculate_monthly_arrival_rates,
    create_abm_config,
    fit_lead_time_distribution,
    fit_wtp_distribution,
)
from .loader import load_runtime_configs
from .schema import (
    ABMConfig,
    EnvConfig,
    LogConfig,
    PathConfig,
    RLConfig,
    RandomConfig,
    SimulationConfig,
    SystemConfig,
)
from .validators import validate_config as _validate_config

__all__ = [
    'PROJECT_ROOT',
    'DATA_PATH',
    'PathConfig',
    'ABMConfig',
    'RLConfig',
    'EnvConfig',
    'SimulationConfig',
    'RandomConfig',
    'SystemConfig',
    'LogConfig',
    'ABM_PERTURBATION_TEMPLATES',
    'apply_abm_perturbation_template',
    'calculate_monthly_arrival_rates',
    'fit_lead_time_distribution',
    'build_empirical_lead_time_distribution',
    'fit_wtp_distribution',
    'create_abm_config',
    'PATH_CONFIG',
    'ABM_CONFIG',
    'ABM_PERTURBATION_TEMPLATE',
    'RL_CONFIG',
    'ENV_CONFIG',
    'SIMULATION_CONFIG',
    'RANDOM_CONFIG',
    'SYSTEM_CONFIG',
    'LOG_CONFIG',
    'validate_config',
]

# 使用更直观的常量切换扰动模板（none / mild / medium / stress）
ABM_PERTURBATION_TEMPLATE = 'mild'

(
    PATH_CONFIG,
    ABM_CONFIG,
    RL_CONFIG,
    ENV_CONFIG,
    SIMULATION_CONFIG,
    RANDOM_CONFIG,
    SYSTEM_CONFIG,
    LOG_CONFIG,
    _runtime_template,
) = load_runtime_configs(perturbation_template=ABM_PERTURBATION_TEMPLATE)


def validate_config() -> bool:
    """兼容旧签名：无参验证当前全局配置。"""
    return _validate_config(RL_CONFIG, ENV_CONFIG)


if not validate_config():
    print("配置验证失败，请检查配置文件")
