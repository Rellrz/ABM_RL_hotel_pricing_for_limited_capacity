from __future__ import annotations

from typing import Any

from configs.config import PPO_CONFIG, SAC_CONFIG
from src.training import train_ppo, train_sac


ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "ppo": {
        "label": "ppo",
        "config": PPO_CONFIG,
        "train_single_run": train_ppo.train_single_run,
        "build_eval_env": train_ppo.build_eval_env,
    },
    "sac": {
        "label": "sac",
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
]
