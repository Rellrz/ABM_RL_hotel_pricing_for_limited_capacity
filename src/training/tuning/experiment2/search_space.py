"""PPO调参搜索空间。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


SPACE_META = [
    ("ppo_learning_rate", "float", "log", 1e-5, 3e-4),
    ("ppo_ent_coef", "float", "log", 1e-4, 2e-2),
    ("ppo_clip_range", "float", "linear", 0.1, 0.4),
    ("ppo_gae_lambda", "float", "linear", 0.9, 0.99),
    ("ppo_gamma", "float", "linear", 0.97, 0.999),
    ("ppo_log_std_init", "float", "linear", -2.0, -0.2),
    ("ppo_shaped_reward_weight", "float", "linear", 0.1, 1.0),
]


N_STEPS_CHOICES = [128, 256, 365, 512, 730]
BATCH_SIZE_CHOICES = [32, 64, 128, 256]
REWARD_MODE_CHOICES = ["scaled_raw_daily", "shaped_bucket", "mixed"]


@dataclass(frozen=True)
class SpaceBounds:
    ent_coef_low: float; ent_coef_high: float
    lr_low: float; lr_high: float
    clip_low: float; clip_high: float
    gae_low: float; gae_high: float
    gamma_low: float; gamma_high: float
    logstd_low: float; logstd_high: float
    shaped_w_low: float; shaped_w_high: float
    n_steps_choices: List[int]
    batch_size_choices: List[int]
    reward_mode_choices: List[str]


GLOBAL_BOUNDS = SpaceBounds(
    ent_coef_low=1e-4, ent_coef_high=2e-2,
    lr_low=1e-5, lr_high=3e-4,
    clip_low=0.1, clip_high=0.4,
    gae_low=0.9, gae_high=0.99,
    gamma_low=0.97, gamma_high=0.999,
    logstd_low=-2.0, logstd_high=-0.2,
    shaped_w_low=0.1, shaped_w_high=1.0,
    n_steps_choices=N_STEPS_CHOICES,
    batch_size_choices=BATCH_SIZE_CHOICES,
    reward_mode_choices=REWARD_MODE_CHOICES,
)

_TUNABLE_SET = frozenset([
    "ppo_learning_rate", "ppo_ent_coef", "ppo_clip_range", "ppo_gae_lambda",
    "ppo_gamma", "ppo_log_std_init", "ppo_shaped_reward_weight",
    "ppo_n_steps", "ppo_batch_size", "ppo_reward_mode",
])


def get_tunable_param_names() -> List[str]:
    return sorted(_TUNABLE_SET)


def suggest_ppo_params(trial, bounds: SpaceBounds | None = None) -> Dict[str, float | int | str]:
    b = bounds or GLOBAL_BOUNDS

    reward_mode = str(trial.suggest_categorical("ppo_reward_mode", list(b.reward_mode_choices)))

    shaped_w = float(trial.suggest_float("ppo_shaped_reward_weight", b.shaped_w_low, b.shaped_w_high))

    params: Dict[str, float | int | str] = {
        "ppo_learning_rate": float(trial.suggest_float("ppo_learning_rate", b.lr_low, b.lr_high, log=True)),
        "ppo_ent_coef": float(trial.suggest_float("ppo_ent_coef", b.ent_coef_low, b.ent_coef_high, log=True)),
        "ppo_clip_range": float(trial.suggest_float("ppo_clip_range", b.clip_low, b.clip_high)),
        "ppo_gae_lambda": float(trial.suggest_float("ppo_gae_lambda", b.gae_low, b.gae_high)),
        "ppo_gamma": float(trial.suggest_float("ppo_gamma", b.gamma_low, b.gamma_high)),
        "ppo_log_std_init": float(trial.suggest_float("ppo_log_std_init", b.logstd_low, b.logstd_high)),
        "ppo_shaped_reward_weight": shaped_w,
        "ppo_n_steps": int(trial.suggest_categorical("ppo_n_steps", list(b.n_steps_choices))),
        "ppo_batch_size": int(trial.suggest_categorical("ppo_batch_size", list(b.batch_size_choices))),
        "ppo_reward_mode": reward_mode,
    }
    return params


def _numeric_range(trials_df: pd.DataFrame, col: str, g_low: float, g_high: float) -> tuple[float, float]:
    cmin = float(trials_df[col].min())
    cmax = float(trials_df[col].max())
    span = max((cmax - cmin) * 0.3, (g_high - g_low) * 0.05)
    low = max(g_low, cmin - span)
    high = min(g_high, cmax + span)
    if high <= low:
        high = min(g_high, low + (g_high - g_low) * 0.1)
    return low, high


def _categorical_refine(col: str, trials_df: pd.DataFrame, global_choices: List) -> List:
    top = trials_df.sort_values("Score", ascending=False).head(max(1, min(6, len(trials_df))))
    seen = sorted(int(x) for x in top[col].dropna().unique() if isinstance(x, (int, float)))
    if not seen:
        return list(global_choices)
    extra = [x for x in seen if x not in global_choices]
    return seen + extra if extra else seen


def build_refine_bounds(trials_df: pd.DataFrame, top_k: int = 6) -> SpaceBounds:
    if trials_df is None or len(trials_df) == 0:
        return GLOBAL_BOUNDS
    use_df = trials_df.copy()
    if "Stable" in use_df.columns:
        stable_df = use_df[use_df["Stable"] == True]
        if len(stable_df) > 0:
            use_df = stable_df
    top_df = use_df.sort_values("Score", ascending=False).head(max(1, int(top_k)))

    ent_low, ent_high = _numeric_range(top_df, "ppo_ent_coef", GLOBAL_BOUNDS.ent_coef_low, GLOBAL_BOUNDS.ent_coef_high)
    lr_low, lr_high = _numeric_range(top_df, "ppo_learning_rate", GLOBAL_BOUNDS.lr_low, GLOBAL_BOUNDS.lr_high)
    clip_low, clip_high = _numeric_range(top_df, "ppo_clip_range", GLOBAL_BOUNDS.clip_low, GLOBAL_BOUNDS.clip_high)
    gae_low, gae_high = _numeric_range(top_df, "ppo_gae_lambda", GLOBAL_BOUNDS.gae_low, GLOBAL_BOUNDS.gae_high)
    gamma_low, gamma_high = _numeric_range(top_df, "ppo_gamma", GLOBAL_BOUNDS.gamma_low, GLOBAL_BOUNDS.gamma_high)
    logstd_low, logstd_high = _numeric_range(top_df, "ppo_log_std_init", GLOBAL_BOUNDS.logstd_low, GLOBAL_BOUNDS.logstd_high)
    shaped_w_low, shaped_w_high = _numeric_range(top_df, "ppo_shaped_reward_weight", GLOBAL_BOUNDS.shaped_w_low, GLOBAL_BOUNDS.shaped_w_high)

    n_steps_choices = _categorical_refine("ppo_n_steps", top_df, GLOBAL_BOUNDS.n_steps_choices)
    batch_size_choices = _categorical_refine("ppo_batch_size", top_df, GLOBAL_BOUNDS.batch_size_choices)
    reward_mode_choices = sorted(top_df["ppo_reward_mode"].dropna().unique().tolist()) or GLOBAL_BOUNDS.reward_mode_choices

    return SpaceBounds(
        ent_coef_low=ent_low, ent_coef_high=ent_high,
        lr_low=lr_low, lr_high=lr_high,
        clip_low=clip_low, clip_high=clip_high,
        gae_low=gae_low, gae_high=gae_high,
        gamma_low=gamma_low, gamma_high=gamma_high,
        logstd_low=logstd_low, logstd_high=logstd_high,
        shaped_w_low=shaped_w_low, shaped_w_high=shaped_w_high,
        n_steps_choices=n_steps_choices,
        batch_size_choices=batch_size_choices,
        reward_mode_choices=reward_mode_choices,
    )
