"""调参报告与论文图表输出。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prep_style() -> None:
    sns.set_context("paper", font_scale=1.25)
    sns.set_style("whitegrid")


def _save_dual(fig, base_path_no_ext: Path, timestamp: str) -> None:
    base_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    output_base = base_path_no_ext.with_name(f"{base_path_no_ext.name}_{timestamp}")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_search_trajectory(trials_df: pd.DataFrame, out_dir: Path, timestamp: str) -> None:
    if trials_df is None or len(trials_df) == 0:
        return
    _prep_style()
    df = trials_df.sort_values("TrialID").copy()
    df["BestSoFar"] = df["Score"].cummax()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(df["TrialID"], df["Score"], marker="o", linewidth=1.2, alpha=0.8, label="Trial Score")
    ax.plot(df["TrialID"], df["BestSoFar"], linewidth=2.0, label="Best So Far")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Objective Score")
    ax.legend()
    _save_dual(fig, out_dir / "search_trajectory", timestamp=timestamp)


def plot_param_sensitivity(trials_df: pd.DataFrame, out_dir: Path, timestamp: str) -> None:
    if trials_df is None or len(trials_df) == 0:
        return
    _prep_style()
    params = [
        "ppo_learning_rate",
        "ppo_ent_coef",
        "ppo_clip_range",
        "ppo_gae_lambda",
        "ppo_gamma",
        "ppo_log_std_init",
        "ppo_shaped_reward_weight",
    ]
    log_params = {"ppo_learning_rate", "ppo_ent_coef"}
    n = len(params)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    for idx, p in enumerate(params):
        ax = axes[idx]
        if p in trials_df.columns:
            sns.scatterplot(data=trials_df, x=p, y="MeanEvalHotelRevenue", hue="ppo_reward_mode", ax=ax, s=30, alpha=0.85)
            if p in log_params:
                ax.set_xscale("log")
            ax.set_xlabel(p)
            ax.set_ylabel("Mean Eval Hotel Revenue")
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    _save_dual(fig, out_dir / "param_sensitivity", timestamp=timestamp)


def plot_pareto(trials_df: pd.DataFrame, out_dir: Path, timestamp: str) -> None:
    if trials_df is None or len(trials_df) == 0:
        return
    _prep_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sns.scatterplot(
        data=trials_df,
        x="TailDrawdown",
        y="MeanEvalHotelRevenue",
        hue="Score",
        style="Stable",
        s=70,
        ax=ax,
        palette="magma",
    )
    ax.set_xlabel("Tail Drawdown")
    ax.set_ylabel("Mean Eval Hotel Revenue")
    _save_dual(fig, out_dir / "stability_performance_pareto", timestamp=timestamp)


def _topk_trial_ids(trials_df: pd.DataFrame, k: int = 5) -> List[int]:
    use = trials_df.copy()
    stable = use[use["Stable"] == True]  # noqa: E712
    if len(stable) > 0:
        use = stable
    return use.sort_values("Score", ascending=False).head(k)["TrialID"].astype(int).tolist()


def plot_topk_learning_curves(trials_df: pd.DataFrame, training_df: pd.DataFrame, out_dir: Path, timestamp: str, k: int = 5) -> None:
    if trials_df is None or training_df is None or len(trials_df) == 0 or len(training_df) == 0:
        return
    top_ids = _topk_trial_ids(trials_df=trials_df, k=k)
    use = training_df[training_df["TrialID"].isin(top_ids)].copy()
    if len(use) == 0:
        return
    _prep_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.lineplot(
        data=use,
        x="Episode",
        y="EpisodeHotelRevenue",
        hue="TrialID",
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Hotel Revenue")
    ax.set_title(f"Top-{k} Trial Learning Curves")
    _save_dual(fig, out_dir / "topk_learning_curves", timestamp=timestamp)


def plot_best_vs_baseline(eval_df: pd.DataFrame, best_trial_id: int, baseline_trial_id: int, out_dir: Path, timestamp: str) -> None:
    if eval_df is None or len(eval_df) == 0:
        return
    use = eval_df[eval_df["TrialID"].isin([int(best_trial_id), int(baseline_trial_id)])].copy()
    if len(use) == 0:
        return
    tag_map = {int(baseline_trial_id): "Baseline", int(best_trial_id): "Best Tuned"}
    use["Group"] = use["TrialID"].map(tag_map)
    _prep_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    metrics = ["EvalHotelRevenue", "EvalOTAProfit", "EvalSystemProfit"]
    labels = ["Hotel Revenue", "OTA Profit", "System Profit"]
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        per_seed = (
            use.groupby(["Group", "Seed"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "SeedMean"})
        )
        agg = per_seed.groupby("Group", as_index=False)["SeedMean"].agg(["mean", "std", "count"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        agg["ci95"] = 1.96 * agg["std"] / np.sqrt(np.maximum(agg["count"], 1))
        ax = axes[idx]
        ax.bar(agg["Group"], agg["mean"], yerr=agg["ci95"], capsize=4, alpha=0.85)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=15)
    _save_dual(fig, out_dir / "best_vs_baseline", timestamp=timestamp)


def plot_drawdown_diagnostics(training_df: pd.DataFrame, best_trial_id: int, out_dir: Path, timestamp: str) -> None:
    if training_df is None or len(training_df) == 0:
        return
    use = training_df[training_df["TrialID"] == int(best_trial_id)].copy()
    if len(use) == 0:
        return
    agg = use.groupby("Episode", as_index=False)["EpisodeHotelRevenue"].mean().sort_values("Episode")
    y = agg["EpisodeHotelRevenue"].to_numpy(dtype=float)
    x = agg["Episode"].to_numpy(dtype=int)
    if y.size == 0:
        return
    roll_w = max(10, int(y.size * 0.05))
    smooth = pd.Series(y).rolling(roll_w, min_periods=1).mean().to_numpy()
    peak_idx = int(np.argmax(smooth))
    tail_start = max(peak_idx, int(y.size * 0.8))

    _prep_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(x, y, alpha=0.25, label="Raw Mean Revenue")
    ax.plot(x, smooth, linewidth=2.0, label=f"Rolling Mean (w={roll_w})")
    ax.axvline(x=x[peak_idx], linestyle="--", linewidth=1.2, label="Peak")
    ax.axvspan(x[tail_start], x[-1], alpha=0.12, color="red", label="Tail Region")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Hotel Revenue")
    ax.legend()
    _save_dual(fig, out_dir / "drawdown_diagnostics", timestamp=timestamp)


def generate_tuning_figures(
    trials_df: pd.DataFrame,
    training_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    best_trial_id: int,
    baseline_trial_id: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_search_trajectory(trials_df=trials_df, out_dir=out_dir, timestamp=timestamp)
    plot_param_sensitivity(trials_df=trials_df, out_dir=out_dir, timestamp=timestamp)
    plot_pareto(trials_df=trials_df, out_dir=out_dir, timestamp=timestamp)
    plot_topk_learning_curves(trials_df=trials_df, training_df=training_df, out_dir=out_dir, timestamp=timestamp, k=5)
    plot_best_vs_baseline(eval_df=eval_df, best_trial_id=best_trial_id, baseline_trial_id=baseline_trial_id, out_dir=out_dir, timestamp=timestamp)
    plot_drawdown_diagnostics(training_df=training_df, best_trial_id=best_trial_id, out_dir=out_dir, timestamp=timestamp)
