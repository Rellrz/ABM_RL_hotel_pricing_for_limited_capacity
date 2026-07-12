from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from datetime import datetime
from itertools import product
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
from src.training.algorithm_registry import get_algorithm_choices, get_algorithm_runner


DEFAULT_CAPACITIES = [20, 30, 40, 50, 60]
DEFAULT_EVAL_SEEDS = [142, 143, 144, 145, 146]
DEFAULT_PRICE_GRID = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
DEFAULT_HEURISTIC_BASES = [
    (50.0, 50.0, 50.0),
    (50.0, 100.0, 150.0),
    (80.0, 120.0, 160.0),
    (100.0, 150.0, 200.0),
]
DEFAULT_HEURISTIC_ALPHA = [0.0, 50.0, 100.0, 150.0]
DEFAULT_HEURISTIC_WEEKEND = [0.0, 20.0, 40.0]
DEFAULT_HEURISTIC_DAY = [0.0, 20.0, 40.0]
METRIC_BASE_NAMES = [
    "episode_revenue",
    "episode_raw_reward",
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
RATIO_METRICS = [
    "learned_vs_hard_upper",
    "learned_vs_static_grid_best",
    "learned_vs_heuristic_best",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学习策略两层评估实验：硬上界 + 强基准")
    parser.add_argument("--algo", type=str, default=None, choices=get_algorithm_choices(), help="单个训练算法")
    parser.add_argument(
        "--algos",
        nargs="+",
        type=str,
        default=None,
        choices=get_algorithm_choices(),
        help="要一起运行并对比的训练算法列表",
    )
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES, help="要扫描的容量列表")
    parser.add_argument("--train-seed", type=int, default=None, help="训练用随机种子，默认使用所选算法配置")
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_EVAL_SEEDS,
        help="评估用随机种子列表，所有策略共享同一组 seed",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="每个容量下学习策略的训练步数，默认使用所选算法配置",
    )
    parser.add_argument(
        "--price-grid",
        nargs="+",
        type=float,
        default=DEFAULT_PRICE_GRID,
        help="静态价格网格搜索使用的价格列表",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="policy_benchmark",
        help="实验运行名前缀",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="关闭 Stable-Baselines3 训练进度条",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并行进程数，设为大于 1 时会对不同 capacity 并行训练与评估",
    )
    return parser.parse_args()


def resolve_algos(args: argparse.Namespace) -> list[str]:
    if args.algos:
        return [str(algo) for algo in args.algos]
    if args.algo:
        return [str(args.algo)]
    return ["ppo_tanh_gaussian"]


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


def evaluate_learned_policy(
    model,
    train_vec_env,
    build_eval_env_fn,
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seed: int,
) -> dict[str, float]:
    eval_env = build_eval_env_fn(
        train_vec_env=train_vec_env,
        historical_data=historical_data,
        seed=eval_seed,
        capacity=capacity,
    )
    aggregator = EpisodeMetricsAggregator()
    episode_reward = 0.0
    obs = eval_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        aggregator.update(infos[0])
        episode_reward += float(rewards[0])
        done = bool(dones[0])

    eval_env.close()
    metrics = aggregator.summary()
    metrics["episode_reward"] = float(episode_reward)
    return metrics


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


def get_heuristic_policy(
    base_prices: tuple[float, float, float],
    scarcity_alpha: float,
    weekend_bonus: float,
    day_premium: float,
    capacity: int,
) -> Callable[[np.ndarray], np.ndarray]:
    base = np.asarray(base_prices, dtype=np.float32)

    def _policy(obs: np.ndarray) -> np.ndarray:
        is_weekday_by_offset = np.asarray(obs[0:3], dtype=np.float32)
        is_weekend_by_offset = 1.0 - is_weekday_by_offset
        inventory = np.asarray(obs[3:6], dtype=np.float32)
        scarcity = 1.0 - inventory / max(1.0, float(capacity))
        prices = base + float(scarcity_alpha) * scarcity + float(weekend_bonus) * is_weekend_by_offset
        prices = prices + np.asarray([0.0, day_premium, 2.0 * day_premium], dtype=np.float32)
        return np.clip(prices, float(ENV_CONFIG.price_min), float(ENV_CONFIG.price_max))

    return _policy


def search_static_grid_best(
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seeds: list[int],
    price_grid: list[float],
) -> tuple[dict[str, float], dict[str, float | str]]:
    best_summary: dict[str, float] | None = None
    best_meta: dict[str, float | str] | None = None
    best_revenue = -np.inf

    for price_tuple in product(price_grid, repeat=3):
        policy_fn = get_static_policy(tuple(map(float, price_tuple)))
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
                "best_static_prices": json.dumps(list(map(float, price_tuple)), ensure_ascii=False),
            }

    assert best_summary is not None and best_meta is not None
    return best_summary, best_meta


def search_heuristic_best(
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seeds: list[int],
) -> tuple[dict[str, float], dict[str, float | str]]:
    best_summary: dict[str, float] | None = None
    best_meta: dict[str, float | str] | None = None
    best_revenue = -np.inf

    heuristic_grid = product(
        DEFAULT_HEURISTIC_BASES,
        DEFAULT_HEURISTIC_ALPHA,
        DEFAULT_HEURISTIC_WEEKEND,
        DEFAULT_HEURISTIC_DAY,
    )
    for base_prices, scarcity_alpha, weekend_bonus, day_premium in heuristic_grid:
        policy_fn = get_heuristic_policy(
            base_prices=tuple(map(float, base_prices)),
            scarcity_alpha=float(scarcity_alpha),
            weekend_bonus=float(weekend_bonus),
            day_premium=float(day_premium),
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
                "heuristic_base_prices": json.dumps(list(map(float, base_prices)), ensure_ascii=False),
                "heuristic_scarcity_alpha": float(scarcity_alpha),
                "heuristic_weekend_bonus": float(weekend_bonus),
                "heuristic_day_premium": float(day_premium),
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


def run_capacity_benchmark_job(
    algos: list[str],
    capacity: int,
    historical_data: pd.DataFrame | None,
    train_seed: int | None,
    eval_seeds: list[int],
    total_timesteps: int | None,
    no_progress_bar: bool,
    run_prefix: str,
    price_grid: list[float],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    if historical_data is None:
        historical_data = load_filtered_historical_data()
    static_summary, static_meta = search_static_grid_best(
        historical_data=historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
        price_grid=list(map(float, price_grid)),
    )
    heuristic_summary, heuristic_meta = search_heuristic_best(
        historical_data=historical_data,
        capacity=int(capacity),
        eval_seeds=list(map(int, eval_seeds)),
    )

    hard_upper_bound = float(ENV_CONFIG.price_max) * float(capacity) * float(ENV_CONFIG.episode_days + 2)
    strategy_rows: list[dict[str, float | int | str]] = [
        make_strategy_row(
            capacity=int(capacity),
            strategy_name="static_grid_best",
            summary=static_summary,
            extra={"algo": "baseline", **static_meta},
        ),
        make_strategy_row(
            capacity=int(capacity),
            strategy_name="heuristic_best",
            summary=heuristic_summary,
            extra={"algo": "baseline", **heuristic_meta},
        ),
    ]
    summary_rows: list[dict[str, float | int | str]] = []

    for algo in algos:
        runner = get_algorithm_runner(algo)
        algo_config = runner["config"]
        train_single_run_fn = runner["train_single_run"]
        build_eval_env_fn = runner["build_eval_env"]
        effective_train_seed = int(algo_config.seed if train_seed is None else train_seed)
        effective_total_timesteps = int(algo_config.total_timesteps if total_timesteps is None else total_timesteps)
        run_name = f"{run_prefix}_{algo}_cap{capacity}"
        model, train_vec_env, run_dir = train_single_run_fn(
            run_name=run_name,
            historical_data=historical_data,
            capacity=int(capacity),
            train_seed=effective_train_seed,
            total_timesteps=effective_total_timesteps,
            progress_bar=not bool(no_progress_bar),
            verbose=1,
        )
        learned_summary, _ = evaluate_policy_over_seeds(
            lambda seed: evaluate_learned_policy(
                model,
                train_vec_env,
                build_eval_env_fn,
                historical_data,
                capacity,
                seed,
            ),
            eval_seeds,
            capacity,
        )
        train_vec_env.close()

        strategy_rows.append(
            make_strategy_row(
                capacity=int(capacity),
                strategy_name=str(algo),
                summary=learned_summary,
                extra={
                    "algo": str(algo),
                    "train_seed": effective_train_seed,
                    "eval_seed_count": int(len(eval_seeds)),
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                },
            )
        )
        summary_rows.append(
            {
                "algo": str(algo),
                "capacity": int(capacity),
                "hard_upper_bound": float(hard_upper_bound),
                "learned_revenue": float(learned_summary["episode_revenue_mean"]),
                "static_grid_best_revenue": float(static_summary["episode_revenue_mean"]),
                "heuristic_best_revenue": float(heuristic_summary["episode_revenue_mean"]),
                "learned_vs_hard_upper": float(learned_summary["episode_revenue_mean"] / max(1e-8, hard_upper_bound)),
                "learned_vs_static_grid_best": float(
                    learned_summary["episode_revenue_mean"] / max(1e-8, static_summary["episode_revenue_mean"])
                ),
                "learned_vs_heuristic_best": float(
                    learned_summary["episode_revenue_mean"] / max(1e-8, heuristic_summary["episode_revenue_mean"])
                ),
                "best_static_prices": str(static_meta["best_static_prices"]),
                "heuristic_base_prices": str(heuristic_meta["heuristic_base_prices"]),
                "heuristic_scarcity_alpha": float(heuristic_meta["heuristic_scarcity_alpha"]),
                "heuristic_weekend_bonus": float(heuristic_meta["heuristic_weekend_bonus"]),
                "heuristic_day_premium": float(heuristic_meta["heuristic_day_premium"]),
                "train_seed": effective_train_seed,
                "total_timesteps": effective_total_timesteps,
            }
        )
    return strategy_rows, summary_rows


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


def plot_summary_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for algo, group in df.groupby("algo"):
        ordered = group.sort_values("capacity")
        plt.plot(ordered["capacity"], ordered[metric], marker="o", linewidth=1.8, label=str(algo))
    plt.xlabel("capacity")
    plt.ylabel(metric)
    plt.title(f"{metric} vs capacity")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=160)
    plt.close()


def plot_full_rate_vs_revenue(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for strategy_name, group in df.groupby("strategy_name"):
        plt.scatter(group["full_day_rate_mean"], group["revenue_per_capacity_day_mean"], s=60, alpha=0.85, label=strategy_name)
        for _, row in group.iterrows():
            plt.annotate(f"cap={int(row['capacity'])}", (row["full_day_rate_mean"], row["revenue_per_capacity_day_mean"]))
    plt.xlabel("full_day_rate_mean")
    plt.ylabel("revenue_per_capacity_day_mean")
    plt.title("full_day_rate vs revenue_per_capacity_day")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "full_day_rate_vs_revenue_per_capacity_day.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    algos = resolve_algos(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    plot_dir = experiment_root / "plots"
    experiment_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    historical_data = load_filtered_historical_data()
    strategy_results: list[dict[str, float | int | str]] = []
    capacity_summaries: list[dict[str, float | int | str]] = []
    capacities = list(map(int, args.capacities))
    eval_seeds = list(map(int, args.eval_seeds))
    price_grid = list(map(float, args.price_grid))
    max_workers = max(1, int(args.max_workers))

    if max_workers == 1:
        for capacity in capacities:
            rows, summaries = run_capacity_benchmark_job(
                algos=algos,
                capacity=int(capacity),
                historical_data=historical_data,
                train_seed=args.train_seed,
                eval_seeds=eval_seeds,
                total_timesteps=args.total_timesteps,
                no_progress_bar=bool(args.no_progress_bar),
                run_prefix=str(args.run_prefix),
                price_grid=price_grid,
            )
            strategy_results.extend(rows)
            capacity_summaries.extend(summaries)
            for summary in summaries:
                print(
                    f"[algo={summary['algo']}, capacity={capacity}] learned={float(summary['learned_revenue']):.2f}, "
                    f"static={float(summary['static_grid_best_revenue']):.2f}, "
                    f"heuristic={float(summary['heuristic_best_revenue']):.2f}"
                )
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_capacity = {
                executor.submit(
                    run_capacity_benchmark_job,
                    algos,
                    int(capacity),
                    None,
                    args.train_seed,
                    eval_seeds,
                    args.total_timesteps,
                    bool(args.no_progress_bar),
                    str(args.run_prefix),
                    price_grid,
                ): int(capacity)
                for capacity in capacities
            }
            for future in as_completed(future_to_capacity):
                capacity = future_to_capacity[future]
                rows, summaries = future.result()
                strategy_results.extend(rows)
                capacity_summaries.extend(summaries)
                for summary in summaries:
                    print(
                        f"[algo={summary['algo']}, capacity={capacity}] learned={float(summary['learned_revenue']):.2f}, "
                        f"static={float(summary['static_grid_best_revenue']):.2f}, "
                        f"heuristic={float(summary['heuristic_best_revenue']):.2f}"
                    )

    strategy_df = pd.DataFrame(strategy_results).sort_values(["strategy_name", "capacity"]).reset_index(drop=True)
    summary_df = pd.DataFrame(capacity_summaries).sort_values(["algo", "capacity"]).reset_index(drop=True)
    strategy_csv = experiment_root / "strategy_results.csv"
    summary_csv = experiment_root / "capacity_summary.csv"
    strategy_df.to_csv(strategy_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    for metric in PLOT_METRICS:
        plot_strategy_metric(strategy_df, metric, plot_dir)
    for metric in RATIO_METRICS:
        plot_summary_metric(summary_df, metric, plot_dir)
    plot_full_rate_vs_revenue(strategy_df, plot_dir)

    summary = {
        "capacities": capacities,
        "algos": algos,
        "train_seed_override": args.train_seed,
        "eval_seeds": eval_seeds,
        "total_timesteps_override": args.total_timesteps,
        "price_grid": price_grid,
        "heuristic_base_options": [list(option) for option in DEFAULT_HEURISTIC_BASES],
        "heuristic_scarcity_alpha_grid": DEFAULT_HEURISTIC_ALPHA,
        "heuristic_weekend_bonus_grid": DEFAULT_HEURISTIC_WEEKEND,
        "heuristic_day_premium_grid": DEFAULT_HEURISTIC_DAY,
        "max_workers": max_workers,
        "strategy_results_csv": str(strategy_csv),
        "capacity_summary_csv": str(summary_csv),
        "plot_dir": str(plot_dir),
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"实验完成，策略结果表: {strategy_csv}")
    print(f"容量汇总表: {summary_csv}")
    print(f"单指标图目录: {plot_dir}")


if __name__ == "__main__":
    main()
