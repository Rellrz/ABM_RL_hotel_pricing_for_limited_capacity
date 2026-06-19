#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict

from .schema import ABMConfig


ABM_PERTURBATION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    'mild': {
        'enable_perturbation': True,
        'demand_ou_theta': 0.18,
        'demand_ou_sigma': 0.04,
        'demand_jump_prob': 0.01,
        'demand_jump_mean': 0.10,
        'demand_jump_std': 0.05,
        'lambda_multiplier_min': 0.80,
        'lambda_multiplier_max': 1.30,
        'wtp_ou_theta': 0.25,
        'wtp_ou_sigma': 0.015,
        'wtp_multiplier_min': 0.92,
        'wtp_multiplier_max': 1.08,
        'wtp_std_multiplier_min': 0.95,
        'wtp_std_multiplier_max': 1.15,
        'channel_pref_ou_theta': 0.30,
        'channel_pref_ou_sigma': 0.02,
        'channel_online_only_prob_min': 0.20,
        'channel_online_only_prob_max': 0.80,
        'utility_noise_type': 'gumbel',
        'utility_gumbel_beta': 0.5,
    },
    'medium': {
        'enable_perturbation': True,
        'demand_ou_theta': 0.15,
        'demand_ou_sigma': 0.08,
        'demand_jump_prob': 0.02,
        'demand_jump_mean': 0.20,
        'demand_jump_std': 0.10,
        'lambda_multiplier_min': 0.60,
        'lambda_multiplier_max': 1.80,
        'wtp_ou_theta': 0.20,
        'wtp_ou_sigma': 0.03,
        'wtp_multiplier_min': 0.85,
        'wtp_multiplier_max': 1.15,
        'wtp_std_multiplier_min': 0.90,
        'wtp_std_multiplier_max': 1.25,
        'channel_pref_ou_theta': 0.25,
        'channel_pref_ou_sigma': 0.04,
        'channel_online_only_prob_min': 0.15,
        'channel_online_only_prob_max': 0.85,
        'utility_noise_type': 'gumbel',
        'utility_gumbel_beta': 0.8,
    },
    'stress': {
        'enable_perturbation': True,
        'demand_ou_theta': 0.10,
        'demand_ou_sigma': 0.14,
        'demand_jump_prob': 0.05,
        'demand_jump_mean': 0.30,
        'demand_jump_std': 0.15,
        'lambda_multiplier_min': 0.45,
        'lambda_multiplier_max': 2.20,
        'wtp_ou_theta': 0.15,
        'wtp_ou_sigma': 0.05,
        'wtp_multiplier_min': 0.75,
        'wtp_multiplier_max': 1.25,
        'wtp_std_multiplier_min': 0.85,
        'wtp_std_multiplier_max': 1.35,
        'channel_pref_ou_theta': 0.20,
        'channel_pref_ou_sigma': 0.07,
        'channel_online_only_prob_min': 0.10,
        'channel_online_only_prob_max': 0.90,
        'utility_noise_type': 'gumbel',
        'utility_gumbel_beta': 1.2,
    },
}


def apply_abm_perturbation_template(cfg: ABMConfig, template_name: str) -> ABMConfig:
    name = str(template_name).strip().lower()
    if name in ('', 'none', 'off', 'false', '0'):
        cfg.enable_perturbation = False
        return cfg

    if name not in ABM_PERTURBATION_TEMPLATES:
        valid = ', '.join(['none'] + sorted(ABM_PERTURBATION_TEMPLATES.keys()))
        raise ValueError(f'Unknown ABM perturbation template: {template_name}. Valid: {valid}')

    template = ABM_PERTURBATION_TEMPLATES[name]
    for k, v in template.items():
        setattr(cfg, k, v)
    return cfg

