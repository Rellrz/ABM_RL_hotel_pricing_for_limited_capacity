"""实验一执行入口。"""

from __future__ import annotations

import argparse
import json

from config import ExperimentConfig
from mdp_solver import MDPSolver
from plotter import ExperimentPlotter
from runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    """提供少量非关键命令行参数，方便调试与复现实验。"""
    parser = argparse.ArgumentParser(description="实验一：最优性与收敛性验证")
    parser.add_argument(
        "--force-recompute-mdp",
        action="store_true",
        help="忽略本地 MDP 缓存并重新估计 P/R。",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="并行 worker 数；不传则使用配置默认值。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig()
    if args.n_jobs is not None:
        config.n_jobs = int(args.n_jobs)
    config.ensure_directories()

    print("=" * 72)
    print("实验一：最优性与收敛性验证")
    print("=" * 72)
    print(f"实验目录: {config.cache_path.parent.parent}")
    print(f"MDP缓存文件: {config.cache_path}")
    print(f"结果CSV: {config.results_csv_path}")
    print(f"结果图: {config.figure_path}")

    solver = MDPSolver(config)
    mdp_result = solver.solve(force_recompute=args.force_recompute_mdp)
    upper_bound = float(mdp_result["upper_bound"])
    print(f"\nMDP Upper Bound V*(s0): {upper_bound:.6f}")

    runner = ExperimentRunner(config)
    df_results, runner_summary = runner.run_all()
    print(f"\n实验结果已保存，共 {len(df_results)} 条记录。")

    plotter = ExperimentPlotter(config)
    plot_summary = plotter.plot(df_results, upper_bound=upper_bound)

    merged_summary = {
        "upper_bound": upper_bound,
        "runner_summary": runner_summary,
        "plot_summary": {
            "figure_path": plot_summary["figure_path"],
            "static_mean": plot_summary["static_mean"],
            "random_mean": plot_summary["random_mean"],
        },
    }
    with open(config.summary_path, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)

    print(f"\n完整闭环已完成。汇总信息已写入: {config.summary_path}")


if __name__ == "__main__":
    main()
