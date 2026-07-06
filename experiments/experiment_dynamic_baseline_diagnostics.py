from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DATA_CONFIG, PATH_CONFIG
from src.baseline.pricing_baselines import (
    DEFAULT_INVENTORY_ALPHA,
    DEFAULT_PRICE_GRID,
    evaluate_manual_policy,
    get_inventory_protection_policy,
    get_static_policy,
    get_weekday_weekend_static_policy,
    rank_static_candidates,
    search_inventory_protection_best,
    search_weekday_weekend_static_best,
    summarize_episode_rows,
)
from src.environment.abm_customer_model import load_eval_historical_data, load_train_historical_data
from src.utils.preprocess_data import data_years_label, get_data_split_metadata


DEFAULT_CAPACITIES = [20, 30, 40, 50, 60]
DEFAULT_EVAL_SEEDS = [142, 143, 144, 145, 146]

PLOT_METRICS = [
    "episode_revenue_mean",
    "revenue_per_capacity_day_mean",
    "full_day_rate_mean",
    "avg_price_day0_mean",
    "avg_price_day1_mean",
    "avg_price_day2_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="动态定价空间诊断 baseline 实验")
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES, help="容量列表")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=DEFAULT_EVAL_SEEDS, help="评估随机种子")
    parser.add_argument("--price-grid", nargs="+", type=float, default=DEFAULT_PRICE_GRID, help="价格网格")
    parser.add_argument(
        "--weekday-weekend-candidates",
        type=int,
        default=20,
        help="工作日/周末静态组合搜索时，从全局静态网格中保留的 top-K 三价候选数",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并行进程数，设为大于 1 时对不同 capacity 并行",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="dynamic_baseline_diagnostics",
        help="输出目录名前缀",
    )
    return parser.parse_args()


def make_strategy_row(
    capacity: int,
    strategy_name: str,
    summary: dict[str, float],
    extra: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "capacity": int(capacity),
        "train_years": data_years_label(DATA_CONFIG.train_years),
        "eval_years": data_years_label(DATA_CONFIG.eval_years),
        "strategy_name": str(strategy_name),
    }
    row.update(summary)
    if extra:
        row.update(extra)
    return row


def run_capacity_job(
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

    ranked_static = rank_static_candidates(
        historical_data=train_historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
        price_grid=list(map(float, price_grid)),
    )
    _, best_static_prices, _ = ranked_static[0]
    candidate_count = max(1, min(int(weekday_weekend_candidates), len(ranked_static)))
    candidate_prices = [prices for _, prices, _ in ranked_static[:candidate_count]]

    _, weekday_weekend_meta = search_weekday_weekend_static_best(
        historical_data=train_historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
        candidate_prices=candidate_prices,
    )
    _, inventory_meta = search_inventory_protection_best(
        historical_data=train_historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
        candidate_prices=candidate_prices,
    )

    static_meta = {"best_static_prices": json.dumps(list(map(float, best_static_prices)), ensure_ascii=False)}
    static_policy = get_static_policy(tuple(map(float, best_static_prices)))
    static_rows = [
        evaluate_manual_policy(static_policy, eval_historical_data, int(capacity), seed)
        for seed in eval_seeds
    ]
    static_summary = summarize_episode_rows(static_rows, int(capacity))
    best_static_revenue = float(static_summary["episode_revenue_mean"])

    weekday_policy = get_weekday_weekend_static_policy(
        weekday_prices=tuple(json.loads(str(weekday_weekend_meta["weekday_static_prices"]))),
        weekend_prices=tuple(json.loads(str(weekday_weekend_meta["weekend_static_prices"]))),
    )
    weekday_weekend_rows = [
        evaluate_manual_policy(weekday_policy, eval_historical_data, int(capacity), seed)
        for seed in eval_seeds
    ]
    weekday_weekend_summary = summarize_episode_rows(weekday_weekend_rows, int(capacity))

    inventory_policy = get_inventory_protection_policy(
        base_prices=tuple(json.loads(str(inventory_meta["inventory_base_prices"]))),
        scarcity_alpha=float(inventory_meta["inventory_scarcity_alpha"]),
        capacity=int(capacity),
    )
    inventory_rows = [
        evaluate_manual_policy(inventory_policy, eval_historical_data, int(capacity), seed)
        for seed in eval_seeds
    ]
    inventory_summary = summarize_episode_rows(inventory_rows, int(capacity))
    strategy_rows = [
        make_strategy_row(
            capacity=int(capacity),
            strategy_name="static_grid_best",
            summary=static_summary,
            extra=static_meta,
        ),
        make_strategy_row(
            capacity=int(capacity),
            strategy_name="weekday_weekend_static_best",
            summary=weekday_weekend_summary,
            extra=weekday_weekend_meta,
        ),
        make_strategy_row(
            capacity=int(capacity),
            strategy_name="inventory_protection_best",
            summary=inventory_summary,
            extra=inventory_meta,
        ),
    ]

    summary_row: dict[str, float | int | str] = {
        "capacity": int(capacity),
        "train_years": data_years_label(DATA_CONFIG.train_years),
        "eval_years": data_years_label(DATA_CONFIG.eval_years),
        "static_grid_best_revenue": float(best_static_revenue),
        "weekday_weekend_static_best_revenue": float(weekday_weekend_summary["episode_revenue_mean"]),
        "inventory_protection_best_revenue": float(inventory_summary["episode_revenue_mean"]),
        "weekday_weekend_vs_static": float(
            weekday_weekend_summary["episode_revenue_mean"] / max(1e-8, best_static_revenue)
        ),
        "inventory_protection_vs_static": float(
            inventory_summary["episode_revenue_mean"] / max(1e-8, best_static_revenue)
        ),
        "best_static_prices": str(static_meta["best_static_prices"]),
        "weekday_static_prices": str(weekday_weekend_meta["weekday_static_prices"]),
        "weekend_static_prices": str(weekday_weekend_meta["weekend_static_prices"]),
        "inventory_base_prices": str(inventory_meta["inventory_base_prices"]),
        "inventory_scarcity_alpha": float(inventory_meta["inventory_scarcity_alpha"]),
        "weekday_weekend_candidate_count": int(candidate_count),
    }
    return strategy_rows, summary_row


def plot_strategy_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for strategy_name, group in df.groupby("strategy_name"):
        ordered = group.sort_values("capacity")
        plt.plot(ordered["capacity"], ordered[metric], marker="o", linewidth=1.8, label=strategy_name)
    plt.xlabel("capacity")
    plt.ylabel(metric)
    plt.title(f"{metric} vs capacity")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=160)
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
    price_grid = list(map(float, args.price_grid))
    max_workers = max(1, int(args.max_workers))
    train_historical_data = load_train_historical_data()
    eval_historical_data = load_eval_historical_data()
    split_metadata = get_data_split_metadata(train_historical_data, eval_historical_data)

    strategy_results: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    if max_workers == 1:
        for capacity in capacities:
            rows, summary = run_capacity_job(
                capacity=int(capacity),
                train_historical_data=train_historical_data,
                eval_historical_data=eval_historical_data,
                eval_seeds=eval_seeds,
                price_grid=price_grid,
                weekday_weekend_candidates=int(args.weekday_weekend_candidates),
            )
            strategy_results.extend(rows)
            summary_rows.append(summary)
            print(
                f"[capacity={capacity}] static={float(summary['static_grid_best_revenue']):.2f}, "
                f"weekday_weekend={float(summary['weekday_weekend_static_best_revenue']):.2f}, "
                f"inventory={float(summary['inventory_protection_best_revenue']):.2f}"
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_capacity = {
                executor.submit(
                    run_capacity_job,
                    int(capacity),
                    None,
                    None,
                    eval_seeds,
                    price_grid,
                    int(args.weekday_weekend_candidates),
                ): int(capacity)
                for capacity in capacities
            }
            for future in as_completed(future_to_capacity):
                capacity = future_to_capacity[future]
                rows, summary = future.result()
                strategy_results.extend(rows)
                summary_rows.append(summary)
                print(
                    f"[capacity={capacity}] static={float(summary['static_grid_best_revenue']):.2f}, "
                    f"weekday_weekend={float(summary['weekday_weekend_static_best_revenue']):.2f}, "
                    f"inventory={float(summary['inventory_protection_best_revenue']):.2f}"
                )

    strategy_df = pd.DataFrame(strategy_results).sort_values(["strategy_name", "capacity"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("capacity").reset_index(drop=True)
    strategy_csv = experiment_root / "strategy_results.csv"
    summary_csv = experiment_root / "capacity_summary.csv"
    strategy_df.to_csv(strategy_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    for metric in PLOT_METRICS:
        plot_strategy_metric(strategy_df, metric, plot_dir)

    metadata = {
        "capacities": capacities,
        "eval_seeds": eval_seeds,
        "price_grid": price_grid,
        "weekday_weekend_candidate_count": int(args.weekday_weekend_candidates),
        "inventory_scarcity_alpha_grid": DEFAULT_INVENTORY_ALPHA,
        **split_metadata,
        "max_workers": max_workers,
        "strategy_results_csv": str(strategy_csv),
        "capacity_summary_csv": str(summary_csv),
        "plot_dir": str(plot_dir),
        "note": "weekday_weekend_static_best uses top-K global static triples as candidate pairs, not exhaustive price_grid^6.",
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"实验完成，策略结果表: {strategy_csv}")
    print(f"容量汇总表: {summary_csv}")
    print(f"单指标图目录: {plot_dir}")


if __name__ == "__main__":
    main()
