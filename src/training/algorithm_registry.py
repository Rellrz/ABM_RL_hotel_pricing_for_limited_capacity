from __future__ import annotations

from typing import Any

from configs.config import PPO_CONFIG, SAC_CONFIG
from src.training import train_ppo, train_sac


def train_ppo_standard(*args, **kwargs):
    return _train_ppo_with_policy_variant("standard", *args, **kwargs)


def train_ppo_tanh_gaussian(*args, **kwargs):
    return _train_ppo_with_policy_variant("tanh_gaussian", *args, **kwargs)


def train_ppo_truncated_gaussian(*args, **kwargs):
    return _train_ppo_with_policy_variant("truncated_gaussian", *args, **kwargs)


def train_ppo_scale_adjusted_truncated_gaussian(*args, **kwargs):
    return _train_ppo_with_policy_variant("scale_adjusted_truncated_gaussian", *args, **kwargs)


def train_ppo_beta(*args, **kwargs):
    return _train_ppo_with_policy_variant("beta", *args, **kwargs)


def _train_ppo_with_policy_variant(policy_variant: str, *args, **kwargs):
    original_policy_variant = str(PPO_CONFIG.policy_variant)
    PPO_CONFIG.policy_variant = str(policy_variant)
    try:
        return train_ppo.train_single_run(*args, **kwargs)
    finally:
        PPO_CONFIG.policy_variant = original_policy_variant


ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "ppo_standard": {
        "label": "ppo_standard",
        "family": "ppo",
        "policy_variant": "standard",
        "default_run_name": "idea2_ppo_standard",
        "config": PPO_CONFIG,
        "train_single_run": train_ppo_standard,
        "build_eval_env": train_ppo.build_eval_env,
    },
    "ppo_tanh_gaussian": {
        "label": "ppo_tanh_gaussian",
        "family": "ppo",
        "policy_variant": "tanh_gaussian",
        "default_run_name": "idea2_ppo_tanh_gaussian",
        "config": PPO_CONFIG,
        "train_single_run": train_ppo_tanh_gaussian,
        "build_eval_env": train_ppo.build_eval_env,
    },
    "ppo_truncated_gaussian": {
        "label": "ppo_truncated_gaussian",
        "family": "ppo",
        "policy_variant": "truncated_gaussian",
        "default_run_name": "idea2_ppo_truncated_gaussian",
        "config": PPO_CONFIG,
        "train_single_run": train_ppo_truncated_gaussian,
        "build_eval_env": train_ppo.build_eval_env,
    },
    "ppo_scale_adjusted_truncated_gaussian": {
        "label": "ppo_scale_adjusted_truncated_gaussian",
        "family": "ppo",
        "policy_variant": "scale_adjusted_truncated_gaussian",
        "default_run_name": "idea2_ppo_scale_adjusted_truncated_gaussian",
        "config": PPO_CONFIG,
        "train_single_run": train_ppo_scale_adjusted_truncated_gaussian,
        "build_eval_env": train_ppo.build_eval_env,
    },
    "ppo_beta": {
        "label": "ppo_beta",
        "family": "ppo",
        "policy_variant": "beta",
        "default_run_name": "idea2_ppo_beta",
        "config": PPO_CONFIG,
        "train_single_run": train_ppo_beta,
        "build_eval_env": train_ppo.build_eval_env,
    },
    "sac": {
        "label": "sac",
        "family": "sac",
        "policy_variant": None,
        "default_run_name": SAC_CONFIG.run_name,
        "config": SAC_CONFIG,
        "train_single_run": train_sac.train_single_run,
        "build_eval_env": train_sac.build_eval_env,
    },
}


def get_algorithm_runner(algo: str) -> dict[str, Any]:
    normalized = str(algo).strip().lower()
    if normalized not in ALGORITHM_REGISTRY:
        supported = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"未知算法: {algo}，当前仅支持: {supported}")
    return ALGORITHM_REGISTRY[normalized]


def get_algorithm_choices() -> list[str]:
    return sorted(ALGORITHM_REGISTRY)


__all__ = [
    "ALGORITHM_REGISTRY",
    "get_algorithm_runner",
    "get_algorithm_choices",
    "train_ppo_standard",
    "train_ppo_tanh_gaussian",
    "train_ppo_truncated_gaussian",
    "train_ppo_scale_adjusted_truncated_gaussian",
    "train_ppo_beta",
]
