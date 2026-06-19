"""实验一的 EJOR 风格可视化模块。"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ExperimentConfig, DEFAULT_CONFIG


class ExperimentPlotter:
    """读取结果并输出论文风格图表与统计表。"""

    def __init__(self, config: ExperimentConfig = DEFAULT_CONFIG):
        self.config = config

    def _summary_table(self, df: pd.DataFrame, upper_bound: float) -> pd.DataFrame:
        """统计最后 100 个 episode 的算法表现。"""
        tail_df = df[df["Episode"] > (self.config.n_episodes - 100)].copy()
        rows = []
        for algorithm in ["CEM", "Static", "Random"]:
            sub = tail_df[tail_df["Algorithm"] == algorithm]
            mean_reward = float(sub["Total_Reward"].mean())
            std_reward = float(sub["Total_Reward"].std(ddof=1))
            optimality_gap = (upper_bound - mean_reward) / upper_bound * 100.0 if upper_bound != 0 else 0.0
            rows.append(
                {
                    "Algorithm": algorithm,
                    "Mean Reward": mean_reward,
                    "Std Dev": std_reward,
                    "Optimality Gap": optimality_gap,
                }
            )
        return pd.DataFrame(rows)

    def _print_markdown_table(self, summary_df: pd.DataFrame) -> None:
        print("\n| Algorithm | Mean Reward | Std Dev | Optimality Gap |")
        print("|---|---:|---:|---:|")
        for _, row in summary_df.iterrows():
            print(
                f"| {row['Algorithm']} | "
                f"{row['Mean Reward']:.4f} | "
                f"{row['Std Dev']:.4f} | "
                f"{row['Optimality Gap']:.2f}% |"
            )

    def plot(self, df: pd.DataFrame, upper_bound: float) -> Dict:
        """绘制收敛图并打印统计表。"""
        sns.set_context("paper", font_scale=1.5)
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))

        cem_df = df[df["Algorithm"] == "CEM"].copy()
        sns.lineplot(
            data=cem_df,
            x="Episode",
            y="Total_Reward",
            estimator="mean",
            errorbar=("ci", 95),
            color="#1f77b4",
            label="CEM",
            ax=ax,
        )

        static_mean = float(df.loc[df["Algorithm"] == "Static", "Total_Reward"].mean())
        random_mean = float(df.loc[df["Algorithm"] == "Random", "Total_Reward"].mean())

        ax.axhline(
            upper_bound,
            color="black",
            linestyle="--",
            linewidth=2.5,
            label="Exact MDP (Upper Bound)",
        )
        ax.axhline(static_mean, color="gray", linewidth=2.0, label="Static")
        ax.axhline(random_mean, color="red", linewidth=2.0, label="Random")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Optimality and Convergence Analysis")
        ax.legend(frameon=True)

        fig.tight_layout()
        fig.savefig(self.config.figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        summary_df = self._summary_table(df, upper_bound)
        self._print_markdown_table(summary_df)

        return {
            "summary_table": summary_df,
            "static_mean": static_mean,
            "random_mean": random_mean,
            "upper_bound": upper_bound,
            "figure_path": str(self.config.figure_path),
        }
