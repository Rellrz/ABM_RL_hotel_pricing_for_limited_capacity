"""绘图：训练期曲线 + 训练后评估柱状图（支持酒店/OTA/系统三类指标）。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.experiment2 import Experiment2Config


def plot_learning_curves(
    config: Experiment2Config,
    training_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    output_path,
) -> None:
    if training_df is None or len(training_df) == 0:
        return
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=training_df,
        x="Episode",
        y=metric_col,
        hue="Algorithm",
        errorbar=("ci", 95),
    )
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_post_eval_bar(
    config: Experiment2Config,
    eval_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    output_path,
) -> None:
    if eval_df is None or len(eval_df) == 0:
        return
    per_seed = (
        eval_df.groupby(["Algorithm", "Seed"], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "SeedMeanMetric"})
    )
    if len(per_seed) == 0:
        return
    agg = (
        per_seed.groupby("Algorithm")["SeedMeanMetric"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "MeanMetric", "std": "StdMetric", "count": "N"})
    )
    agg["StdMetric"] = agg["StdMetric"].fillna(0.0)
    agg["ErrorBar95CI"] = 1.96 * agg["StdMetric"] / np.sqrt(np.maximum(agg["N"], 1))

    plt.figure(figsize=(8, 5))
    x = np.arange(len(agg))
    plt.bar(x, agg["MeanMetric"].values, yerr=agg["ErrorBar95CI"].values, capsize=4, alpha=0.85)
    plt.xticks(x, agg["Algorithm"].values, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("Algorithm")

    # Y 轴自适应数据范围，不从 0 开始
    y_min = (agg["MeanMetric"] - agg["ErrorBar95CI"]).min()
    y_max = (agg["MeanMetric"] + agg["ErrorBar95CI"]).max()
    y_margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
