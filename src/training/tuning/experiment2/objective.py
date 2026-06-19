"""PPO调参目标函数与指标计算。"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _tail_drawdown_for_seed(seed_df: pd.DataFrame) -> float:
    seq = seed_df.sort_values("Episode")["EpisodeHotelRevenue"].to_numpy(dtype=float)
    if seq.size < 20:
        return 0.0
    peak = float(np.max(seq))
    if peak <= 1e-9:
        return 0.0
    tail_len = max(5, int(seq.size * 0.2))
    tail_mean = float(np.mean(seq[-tail_len:]))
    return max(0.0, (peak - tail_mean) / peak)


def summarize_trial_metrics(training_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, float]:
    if training_df is None or len(training_df) == 0:
        return {
            "MeanEvalHotelRevenue": 0.0,
            "StdEvalHotelRevenue": 0.0,
            "TrainTailMean": 0.0,
            "TailDrawdown": 1.0,
            "Score": -1e18,
            "Stable": False,
        }

    if eval_df is None or len(eval_df) == 0:
        mean_eval = 0.0
        std_eval = 0.0
    else:
        per_seed_eval = (
            eval_df.groupby("Seed", as_index=False)["EvalHotelRevenue"]
            .mean()
            .rename(columns={"EvalHotelRevenue": "SeedEvalMean"})
        )
        mean_eval = float(per_seed_eval["SeedEvalMean"].mean())
        std_eval = float(per_seed_eval["SeedEvalMean"].std(ddof=0)) if len(per_seed_eval) > 1 else 0.0

    per_seed_drawdown = []
    per_seed_tail_mean = []
    for _, sdf in training_df.groupby("Seed"):
        per_seed_drawdown.append(_tail_drawdown_for_seed(sdf))
        seq = sdf.sort_values("Episode")["EpisodeHotelRevenue"].to_numpy(dtype=float)
        tail_len = max(5, int(seq.size * 0.2))
        per_seed_tail_mean.append(float(np.mean(seq[-tail_len:])))

    tail_drawdown = float(np.mean(per_seed_drawdown)) if per_seed_drawdown else 1.0
    train_tail_mean = float(np.mean(per_seed_tail_mean)) if per_seed_tail_mean else 0.0

    # 稳定优先：高收益 + 低回撤 + 低波动
    # drawdown_penalty: 回撤惩罚力度，10% 回撤 → 折损 20% 得分
    drawdown_penalty = 2.0
    cv = max(0.0, std_eval / max(1.0, mean_eval)) if mean_eval > 0 else 0.0
    score = mean_eval * (1.0 - drawdown_penalty * tail_drawdown - 0.5 * cv)
    stable = bool(tail_drawdown <= 0.05)
    if not stable:
        score *= 0.9

    return {
        "MeanEvalHotelRevenue": mean_eval,
        "StdEvalHotelRevenue": std_eval,
        "TrainTailMean": train_tail_mean,
        "TailDrawdown": tail_drawdown,
        "Score": score,
        "Stable": stable,
    }


def summarize_trial(
    trial_id: int,
    stage: str,
    params: Dict[str, float | int | str],
    training_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Tuple[Dict, Dict[str, float]]:
    metrics = summarize_trial_metrics(training_df=training_df, eval_df=eval_df)
    row = {
        "TrialID": int(trial_id),
        "Stage": str(stage),
        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in params.items()},
        **metrics,
    }
    row["ppo_n_steps"] = int(params.get("ppo_n_steps", 256))
    row["ppo_batch_size"] = int(params.get("ppo_batch_size", 64))
    return row, metrics

