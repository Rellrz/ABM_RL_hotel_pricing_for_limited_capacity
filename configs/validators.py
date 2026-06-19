#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .schema import ABMConfig, EnvConfig, RLConfig


def validate_config(rl_config: RLConfig, env_config: EnvConfig) -> bool:
    if rl_config.cem_algorithm not in ("cem", "cem_nn"):
        print("错误：cem_algorithm必须为'cem'或'cem_nn'")
        return False

    if not (0.0 <= rl_config.cem_elite_frac <= 1.0):
        print("错误：cem_elite_frac必须在0和1之间")
        return False

    if not (0.0 <= rl_config.cem_alpha <= 1.0):
        print("错误：cem_alpha必须在0和1之间")
        return False

    if rl_config.cem_n_samples <= 0:
        print("错误：cem_n_samples必须大于0")
        return False

    if rl_config.cem_init_strategy not in ("midpoint", "emsrb", "blended"):
        print("错误：cem_init_strategy必须为'midpoint'、'emsrb'或'blended'")
        return False

    if not (0.0 <= rl_config.cem_init_blend_alpha <= 1.0):
        print("错误：cem_init_blend_alpha必须在0和1之间")
        return False

    if rl_config.initial_std <= 0 or rl_config.min_std <= 0:
        print("错误：initial_std和min_std必须大于0")
        return False

    if rl_config.commission_rate < 0 or rl_config.commission_rate > 1:
        print("错误：commission_rate必须在0和1之间")
        return False

    if rl_config.subsidy_ratio_min < 0 or rl_config.subsidy_ratio_max > 1:
        print("错误：补贴比例必须在0和1之间")
        return False

    if rl_config.subsidy_ratio_min > rl_config.subsidy_ratio_max:
        print("错误：subsidy_ratio_min不能大于subsidy_ratio_max")
        return False

    if rl_config.ota_delta_max < 0:
        print("错误：ota_delta_max不能小于0")
        return False

    if rl_config.ota_decay_lambda < 0:
        print("错误：ota_decay_lambda不能小于0")
        return False

    if rl_config.ota_noise_std < 0:
        print("错误：ota_noise_std不能小于0")
        return False

    if rl_config.online_price_min <= 0 or rl_config.offline_price_min <= 0:
        print("错误：最低价格必须大于0")
        return False

    if rl_config.online_price_min > rl_config.online_price_max:
        print("错误：online_price_min不能大于online_price_max")
        return False

    if rl_config.offline_price_min > rl_config.offline_price_max:
        print("错误：offline_price_min不能大于offline_price_max")
        return False

    if env_config.initial_inventory <= 0:
        print("错误：初始库存必须大于0")
        return False

    if env_config.booking_window_days <= 0:
        print("错误：booking_window_days必须大于0")
        return False

    return True


def validate_abm_config(abm_config: ABMConfig) -> bool:
    if abm_config.room_marginal_cost < 0:
        print("错误：room_marginal_cost不能小于0")
        return False

    if not (0.0 <= abm_config.online_discount_ratio <= 1.0):
        print("错误：online_discount_ratio必须在0和1之间")
        return False

    if not (0.0 <= abm_config.anchor_quantile_low <= 1.0 and 0.0 <= abm_config.anchor_quantile_high <= 1.0):
        print("错误：anchor_quantile_low和anchor_quantile_high必须在0和1之间")
        return False

    if abm_config.anchor_quantile_low > abm_config.anchor_quantile_high:
        print("错误：anchor_quantile_low不能大于anchor_quantile_high")
        return False

    if min(abm_config.anchor_weight_low, abm_config.anchor_weight_mean, abm_config.anchor_weight_high) < 0:
        print("错误：anchor权重不能小于0")
        return False

    if abm_config.anchor_eta < 0:
        print("错误：anchor强度参数不能小于0")
        return False

    if min(abm_config.anchor_lambda_plus, abm_config.anchor_lambda_minus) < 0:
        print("错误：anchor_lambda_plus和anchor_lambda_minus不能小于0")
        return False

    if not (0.0 <= abm_config.anchor_joint_theta <= 1.0):
        print("错误：anchor_joint_theta必须在0和1之间")
        return False

    return True
