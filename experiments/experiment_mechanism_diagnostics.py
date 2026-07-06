from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ABM_CONFIG, PATH_CONFIG
from src.environment.abm_customer_model import load_eval_historical_data, load_train_historical_data
from src.utils.preprocess_data import get_data_split_metadata
from experiments.experiment_dynamic_baseline_diagnostics import (
    DEFAULT_PRICE_GRID,
    PLOT_METRICS,
    plot_strategy_metric,
    run_capacity_job,
)


DEFAULT_CAPACITIES = [20, 30, 50]
DEFAULT_EVAL_SEEDS = [142, 143, 144]
DEFAULT_FLEXIBLE_CUSTOMER_SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_LAMBDA_DAY_MISMATCH_FLEX = [6.0, 12.0, 24.0, 48.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="idea2 机制强度诊断实验")
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES, help="容量列表")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=DEFAULT_EVAL_SEEDS, help="评估随机种子")
    parser.add_argument(
        "--flexible-customer-shares",
        nargs="+",
        type=float,
        default=DEFAULT_FLEXIBLE_CUSTOMER_SHARES,
        help="灵活型消费者占比网格",
    )
    parser.add_argument(
        "--lambda-day-mismatch-flex",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDA_DAY_MISMATCH_FLEX,
        help="灵活型消费者日期错配惩罚网格",
    )
    parser.add_argument("--price-grid", nargs="+", type=float, default=DEFAULT_PRICE_GRID, help="价格网格")
    parser.add_argument(
        "--weekday-weekend-candidates",
        type=int,
        default=10,
        help="工作日/周末静态组合搜索时，从全局静态网格中保留的 top-K 三价候选数",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并行进程数，设为大于 1 时对不同机制场景并行",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="mechanism_diagnostics",
        help="输出目录名前缀",
    )
    return parser.parse_args()


@contextmanager
def apply_abm_overrides(flexible_customer_share: float, lambda_day_mismatch_flex: float):
    original_flexible_customer_share = float(ABM_CONFIG.flexible_customer_share)
    original_lambda_day_mismatch_flex = float(ABM_CONFIG.lambda_day_mismatch_flex)
    try:
        ABM_CONFIG.flexible_customer_share = float(flexible_customer_share)
        ABM_CONFIG.lambda_day_mismatch_flex = float(lambda_day_mismatch_flex)
        yield
    finally:
        ABM_CONFIG.flexible_customer_share = original_flexible_customer_share
        ABM_CONFIG.lambda_day_mismatch_flex = original_lambda_day_mismatch_flex


def run_mechanism_job(
    flexible_customer_share: float,
    lambda_day_mismatch_flex: float,
    capacity: int,
    train_historical_data: pd.DataFrame | None,
    eval_historical_data: pd.DataFrame | None,
    eval_seeds: list[int],
    price_grid: list[float],
    weekday_weekend_candidates: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    if train_historical_data is None:
        train_historical_data = load_train_historical_data()
    if eval_historical_data is None:
        eval_historical_data = load_eval_historical_data()

    with apply_abm_overrides(
        flexible_customer_share=float(flexible_customer_share),
        lambda_day_mismatch_flex=float(lambda_day_mismatch_flex),
    ):
        strategy_rows, summary = run_capacity_job(
            capacity=int(capacity),
            train_historical_data=train_historical_data,
            eval_historical_data=eval_historical_data,
            eval_seeds=list(map(int, eval_seeds)),
            price_grid=list(map(float, price_grid)),
            weekday_weekend_candidates=int(weekday_weekend_candidates),
        )

    mechanism_fields: dict[str, float | int | str] = {
        "flexible_customer_share": float(flexible_customer_share),
        "lambda_day_mismatch_flex": float(lambda_day_mismatch_flex),
    }
    for row in strategy_rows:
        row.update(mechanism_fields)
    summary.update(mechanism_fields)
    return strategy_rows, summary


def plot_ratio_heatmaps(summary_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        "weekday_weekend_vs_static",
        "inventory_protection_vs_static",
    ]
    for capacity, capacity_group in summary_df.groupby("capacity"):
        for metric in metrics:
            pivot = capacity_group.pivot(
                index="lambda_day_mismatch_flex",
                columns="flexible_customer_share",
                values=metric,
            ).sort_index(ascending=True)
            plt.figure(figsize=(7, 4.5))
            image = plt.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
            plt.colorbar(image, label=metric)
            plt.xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
            plt.xlabel("flexible_customer_share")
            plt.ylabel("lambda_day_mismatch_flex")
            plt.title(f"{metric}, capacity={int(capacity)}")
            plt.tight_layout()
            plt.savefig(output_dir / f"{metric}_capacity{int(capacity)}.png", dpi=160)
            plt.close()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    plot_dir = experiment_root / "plots"
    experiment_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    capacities = list(map(int, args.capacities))
    eval_seeds = list(map(int, args.eval_seeds))
    flexible_customer_shares = list(map(float, args.flexible_customer_shares))
    lambda_day_mismatch_flex_values = list(map(float, args.lambda_day_mismatch_flex))
    price_grid = list(map(float, args.price_grid))
    max_workers = max(1, int(args.max_workers))
    weekday_weekend_candidates = int(args.weekday_weekend_candidates)
    train_historical_data = load_train_historical_data()
    eval_historical_data = load_eval_historical_data()
    split_metadata = get_data_split_metadata(train_historical_data, eval_historical_data)

    jobs = [
        (flexible_customer_share, lambda_day_mismatch_flex, capacity)
        for flexible_customer_share, lambda_day_mismatch_flex, capacity in product(
            flexible_customer_shares,
            lambda_day_mismatch_flex_values,
            capacities,
        )
    ]
    strategy_results: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    if max_workers == 1:
        for flexible_customer_share, lambda_day_mismatch_flex, capacity in jobs:
            rows, summary = run_mechanism_job(
                flexible_customer_share=float(flexible_customer_share),
                lambda_day_mismatch_flex=float(lambda_day_mismatch_flex),
                capacity=int(capacity),
                train_historical_data=train_historical_data,
                eval_historical_data=eval_historical_data,
                eval_seeds=eval_seeds,
                price_grid=price_grid,
                weekday_weekend_candidates=weekday_weekend_candidates,
            )
            strategy_results.extend(rows)
            summary_rows.append(summary)
            print(
                f"[flex={flexible_customer_share}, lambda_flex={lambda_day_mismatch_flex}, cap={capacity}] "
                f"static={float(summary['static_grid_best_revenue']):.2f}, "
                f"weekday_weekend_ratio={float(summary['weekday_weekend_vs_static']):.4f}, "
                f"inventory_ratio={float(summary['inventory_protection_vs_static']):.4f}"
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    run_mechanism_job,
                    float(flexible_customer_share),
                    float(lambda_day_mismatch_flex),
                    int(capacity),
                    None,
                    None,
                    eval_seeds,
                    price_grid,
                    weekday_weekend_candidates,
                ): (flexible_customer_share, lambda_day_mismatch_flex, capacity)
                for flexible_customer_share, lambda_day_mismatch_flex, capacity in jobs
            }
            for future in as_completed(future_to_job):
                flexible_customer_share, lambda_day_mismatch_flex, capacity = future_to_job[future]
                rows, summary = future.result()
                strategy_results.extend(rows)
                summary_rows.append(summary)
                print(
                    f"[flex={flexible_customer_share}, lambda_flex={lambda_day_mismatch_flex}, cap={capacity}] "
                    f"static={float(summary['static_grid_best_revenue']):.2f}, "
                    f"weekday_weekend_ratio={float(summary['weekday_weekend_vs_static']):.4f}, "
                    f"inventory_ratio={float(summary['inventory_protection_vs_static']):.4f}"
                )

    strategy_df = (
        pd.DataFrame(strategy_results)
        .sort_values(["flexible_customer_share", "lambda_day_mismatch_flex", "capacity", "strategy_name"])
        .reset_index(drop=True)
    )
    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["flexible_customer_share", "lambda_day_mismatch_flex", "capacity"])
        .reset_index(drop=True)
    )

    strategy_csv = experiment_root / "strategy_results.csv"
    summary_csv = experiment_root / "capacity_summary.csv"
    strategy_df.to_csv(strategy_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    for metric in PLOT_METRICS:
        plot_strategy_metric(strategy_df, metric, plot_dir)
    plot_ratio_heatmaps(summary_df, plot_dir)

    metadata = {
        "capacities": capacities,
        "eval_seeds": eval_seeds,
        "flexible_customer_shares": flexible_customer_shares,
        "lambda_day_mismatch_flex": lambda_day_mismatch_flex_values,
        "price_grid": price_grid,
        "weekday_weekend_candidate_count": weekday_weekend_candidates,
        **split_metadata,
        "max_workers": max_workers,
        "strategy_results_csv": str(strategy_csv),
        "capacity_summary_csv": str(summary_csv),
        "plot_dir": str(plot_dir),
        "note": "This experiment does not train RL policies; it diagnoses dynamic pricing room under ABM mechanism scenarios.",
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"实验完成，策略结果表: {strategy_csv}")
    print(f"容量汇总表: {summary_csv}")
    print(f"图表目录: {plot_dir}")


if __name__ == "__main__":
    main()
