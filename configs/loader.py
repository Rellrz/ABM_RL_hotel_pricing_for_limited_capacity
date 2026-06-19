#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

from .base import DATA_PATH
from .defaults import apply_abm_perturbation_template
from .estimators import create_abm_config
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


def load_runtime_configs(
    data_path: Optional[str] = None,
    perturbation_template: Optional[str] = None,
) -> Tuple[PathConfig, ABMConfig, RLConfig, EnvConfig, SimulationConfig, RandomConfig, SystemConfig, LogConfig, str]:
    path_cfg = PathConfig()
    use_data_path = DATA_PATH if data_path is None else data_path
    abm_cfg = create_abm_config(use_data_path)

    template = 'none' if perturbation_template is None else str(perturbation_template)
    abm_cfg = apply_abm_perturbation_template(abm_cfg, template)

    rl_cfg = RLConfig()
    env_cfg = EnvConfig()
    sim_cfg = SimulationConfig()
    rnd_cfg = RandomConfig()
    sys_cfg = SystemConfig()
    log_cfg = LogConfig()
    return path_cfg, abm_cfg, rl_cfg, env_cfg, sim_cfg, rnd_cfg, sys_cfg, log_cfg, template
