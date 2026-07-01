from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ENV_CONFIG, PATH_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data
from src.environment.gym_hotel_env import GymHotelPricingEnv
from src.training.train_ppo import EpisodeMetricsAggregator


DEFAULT_CAPACITIES = [20, 30, 40, 50, 60]
DEFAULT_EVAL_SEEDS = [142, 143, 144, 145, 146]
DEFAULT_PRICE_GRID = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
DEFAULT_INVENTORY_ALPHA = [0.0, 50.0, 100.0, 150.0, 200.0]

METRIC_BASE_NAMES = [
    "episode_revenue",
    "episode_reward",
    "episode_penalty",
    "episode_acceptance_rate",
    "avg_price_day0",
    "avg_price_day1",
    "avg_price_day2",
    "avg_inventory_day0",
    "avg_inventory_day1",
    "avg_inventory_day2",
    "full_day_rate",
    "full_rate_day0",
    "full_rate_day1",
    "full_rate_day2",
    "revenue_per_arrival",
    "revenue_per_capacity_day",
]

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


def prices_to_normalized_action(prices: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (float(ENV_CONFIG.price_min) + float(ENV_CONFIG.price_max))
    half_range = 0.5 * (float(ENV_CONFIG.price_max) - float(ENV_CONFIG.price_min))
    normalized = (np.asarray(prices, dtype=np.float32) - midpoint) / max(1e-8, half_range)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def add_derived_metrics(metrics: dict[str, float], capacity: int) -> dict[str, float]:
    enriched = dict(metrics)
    enriched["revenue_per_arrival"] = float(enriched["episode_revenue"] / max(1.0, enriched["episode_arrivals"]))
    enriched["revenue_per_capacity_day"] = float(
        enriched["episode_revenue"] / max(1.0, float(capacity) * float(ENV_CONFIG.episode_days))
    )
    return enriched


def summarize_episode_rows(rows: list[dict[str, float]], capacity: int) -> dict[str, float]:
    enriched_rows = [add_derived_metrics(row, capacity) for row in rows]
    frame = pd.DataFrame(enriched_rows)
    summary: dict[str, float] = {}
    for metric in METRIC_BASE_NAMES:
        summary[f"{metric}_mean"] = float(frame[metric].mean())
        summary[f"{metric}_std"] = float(frame[metric].std(ddof=0))
    return summary


def evaluate_manual_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seed: int,
) -> dict[str, float]:
    env = GymHotelPricingEnv(
        historical_data=historical_data,
        seed=eval_seed,
        capacity=capacity,
    )
    aggregator = EpisodeMetricsAggregator()
    episode_reward = 0.0
    obs, _ = env.reset(seed=eval_seed)
    done = False

    while not done:
        prices = np.asarray(policy_fn(obs), dtype=np.float32).reshape(3)
        action = prices_to_normalized_action(prices)
        obs, reward, terminated, truncated, info = env.step(action)
        aggregator.update(info)
        episode_reward += float(reward)
        done = bool(terminated or truncated)

    metrics = aggregator.summary()
    metrics["episode_reward"] = float(episode_reward)
    env.close()
    return metrics


def evaluate_policy_over_seeds(
    eval_fn: Callable[[int], dict[str, float]],
    eval_seeds: list[int],
    capacity: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = [eval_fn(seed) for seed in eval_seeds]
    return summarize_episode_rows(rows, capacity), rows


def get_static_policy(prices: tuple[float, float, float]) -> Callable[[np.ndarray], np.ndarray]:
    fixed_prices = np.asarray(prices, dtype=np.float32)

    def _policy(_obs: np.ndarray) -> np.ndarray:
        return fixed_prices

    return _policy


def get_weekday_weekend_static_policy(
    weekday_prices: tuple[float, float, float],
    weekend_prices: tuple[float, float, float],
) -> Callable[[np.ndarray], np.ndarray]:
    weekday = np.asarray(weekday_prices, dtype=np.float32)
    weekend = np.asarray(weekend_prices, dtype=np.float32)

    def _policy(obs: np.ndarray) -> np.ndarray:
        is_weekend = bool(float(obs[1]) >= 0.5)
        return weekend if is_weekend else weekday

    return _policy


def get_inventory_protection_policy(
    base_prices: tuple[float, float, float],
    scarcity_alpha: float,
    capacity: int,
) -> Callable[[np.ndarray], np.ndarray]:
    base = np.asarray(base_prices, dtype=np.float32)

    def _policy(obs: np.ndarray) -> np.ndarray:
        inventory = np.asarray(obs[2:5], dtype=np.float32)
        scarcity = 1.0 - inventory / max(1.0, float(capacity))
        prices = base + float(scarcity_alpha) * scarcity
        return np.clip(prices, float(ENV_CONFIG.price_min), float(ENV_CONFIG.price_max))

    return _policy


def rank_static_candidates(
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seeds: list[int],
    price_grid: list[float],
) -> list[tuple[float, tuple[float, float, float], dict[str, float]]]:
    ranked: list[tuple[float, tuple[float, float, float], dict[str, float]]] = []
    for price_tuple in product(price_grid, repeat=3):
        prices = tuple(map(float, price_tuple))
        policy_fn = get_static_policy(prices)
        summary, _ = evaluate_policy_over_seeds(
            lambda seed: evaluate_manual_policy(policy_fn, historical_data, capacity, seed),
            eval_seeds,
            capacity,
        )
        ranked.append((float(summary["episode_revenue_mean"]), prices, summary))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked


def search_weekday_weekend_static_best(
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seeds: list[int],
    candidate_prices: list[tuple[float, float, float]],
) -> tuple[dict[str, float], dict[str, float | str]]:
    best_summary: dict[str, float] | None = None
    best_meta: dict[str, float | str] | None = None
    best_revenue = -np.inf

    for weekday_prices in candidate_prices:
        for weekend_prices in candidate_prices:
            policy_fn = get_weekday_weekend_static_policy(
                weekday_prices=weekday_prices,
                weekend_prices=weekend_prices,
            )
            summary, _ = evaluate_policy_over_seeds(
                lambda seed: evaluate_manual_policy(policy_fn, historical_data, capacity, seed),
                eval_seeds,
                capacity,
            )
            revenue = float(summary["episode_revenue_mean"])
            if revenue > best_revenue:
                best_revenue = revenue
                best_summary = summary
                best_meta = {
                    "weekday_static_prices": json.dumps(list(map(float, weekday_prices)), ensure_ascii=False),
                    "weekend_static_prices": json.dumps(list(map(float, weekend_prices)), ensure_ascii=False),
                }

    assert best_summary is not None and best_meta is not None
    return best_summary, best_meta


def search_inventory_protection_best(
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seeds: list[int],
    candidate_prices: list[tuple[float, float, float]],
) -> tuple[dict[str, float], dict[str, float | str]]:
    best_summary: dict[str, float] | None = None
    best_meta: dict[str, float | str] | None = None
    best_revenue = -np.inf

    for base_prices, scarcity_alpha in product(candidate_prices, DEFAULT_INVENTORY_ALPHA):
        policy_fn = get_inventory_protection_policy(
            base_prices=base_prices,
            scarcity_alpha=float(scarcity_alpha),
            capacity=capacity,
        )
        summary, _ = evaluate_policy_over_seeds(
            lambda seed: evaluate_manual_policy(policy_fn, historical_data, capacity, seed),
            eval_seeds,
            capacity,
        )
        revenue = float(summary["episode_revenue_mean"])
        if revenue > best_revenue:
            best_revenue = revenue
            best_summary = summary
            best_meta = {
                "inventory_base_prices": json.dumps(list(map(float, base_prices)), ensure_ascii=False),
                "inventory_scarcity_alpha": float(scarcity_alpha),
            }

    assert best_summary is not None and best_meta is not None
    return best_summary, best_meta


def make_strategy_row(
    capacity: int,
    strategy_name: str,
    summary: dict[str, float],
    extra: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "capacity": int(capacity),
        "strategy_name": str(strategy_name),
    }
    row.update(summary)
    if extra:
        row.update(extra)
    return row


def run_capacity_job(
    capacity: int,
    historical_data: pd.DataFrame | None,
    eval_seeds: list[int],
    price_grid: list[float],
    weekday_weekend_candidates: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    if historical_data is None:
        historical_data = load_filtered_historical_data()

    ranked_static = rank_static_candidates(
        historical_data=historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
        price_grid=list(map(float, price_grid)),
    )
    best_static_revenue, best_static_prices, static_summary = ranked_static[0]
    candidate_count = max(1, min(int(weekday_weekend_candidates), len(ranked_static)))
    candidate_prices = [prices for _, prices, _ in ranked_static[:candidate_count]]

    weekday_weekend_summary, weekday_weekend_meta = search_weekday_weekend_static_best(
        historical_data=historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
        candidate_prices=candidate_prices,
    )
    inventory_summary, inventory_meta = search_inventory_protection_best(
        historical_data=historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
        candidate_prices=candidate_prices,
    )

    static_meta = {"best_static_prices": json.dumps(list(map(float, best_static_prices)), ensure_ascii=False)}
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
    historical_data = load_filtered_historical_data()

    strategy_results: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    if max_workers == 1:
        for capacity in capacities:
            rows, summary = run_capacity_job(
                capacity=int(capacity),
                historical_data=historical_data,
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
