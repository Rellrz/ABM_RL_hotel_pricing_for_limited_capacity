from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DATA_CONFIG, PATH_CONFIG
from src.environment.abm_customer_model import load_eval_historical_data, load_train_historical_data
from src.utils.preprocess_data import data_years_label, get_data_split_metadata
from src.training.train_ppo import EpisodeMetricsAggregator
from src.training.algorithm_registry import get_algorithm_choices, get_algorithm_runner


DEFAULT_CAPACITIES = [20, 30, 40, 50, 60]
PLOT_METRICS = [
    "episode_reward",
    "episode_revenue",
    "episode_acceptance_rate",
    "full_day_rate",
    "full_slot_rate",
    "full_rate_day0",
    "full_rate_day1",
    "full_rate_day2",
    "avg_price_day0",
    "avg_price_day1",
    "avg_price_day2",
    "avg_inventory_day0",
    "avg_inventory_day1",
    "avg_inventory_day2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="容量敏感性分析实验")
    parser.add_argument("--algo", type=str, default="ppo_tanh_gaussian", choices=get_algorithm_choices(), help="训练算法")
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES, help="要扫描的容量列表")
    parser.add_argument("--train-seed", type=int, default=None, help="训练用随机种子，默认使用所选算法配置")
    parser.add_argument("--eval-seed", type=int, default=None, help="评估用新随机种子，默认使用 train_seed + 100")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="每个容量的训练步数，默认使用所选算法配置",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="capacity_sensitivity",
        help="训练 run 名前缀，同时用于输出目录命名",
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


def evaluate_policy(
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


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(df["capacity"], df[metric], marker="o", linewidth=1.8)
    plt.xlabel("capacity")
    plt.ylabel(metric)
    plt.title(f"{metric} vs capacity")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=160)
    plt.close()


def run_capacity_job(
    algo: str,
    capacity: int,
    train_historical_data: pd.DataFrame | None,
    eval_historical_data: pd.DataFrame | None,
    train_seed: int,
    eval_seed: int,
    total_timesteps: int,
    no_progress_bar: bool,
    run_prefix: str,
) -> dict[str, float | int | str]:
    if train_historical_data is None:
        train_historical_data = load_train_historical_data()
    if eval_historical_data is None:
        eval_historical_data = load_eval_historical_data()
    runner = get_algorithm_runner(algo)
    train_single_run_fn = runner["train_single_run"]
    build_eval_env_fn = runner["build_eval_env"]
    run_name = f"{run_prefix}_{algo}_cap{capacity}"
    model, train_vec_env, run_dir = train_single_run_fn(
        run_name=run_name,
        historical_data=train_historical_data,
        capacity=int(capacity),
        train_seed=int(train_seed),
        total_timesteps=int(total_timesteps),
        progress_bar=not bool(no_progress_bar),
        verbose=1,
    )
    metrics = evaluate_policy(
        model=model,
        train_vec_env=train_vec_env,
        build_eval_env_fn=build_eval_env_fn,
        historical_data=eval_historical_data,
        capacity=int(capacity),
        eval_seed=int(eval_seed),
    )
    train_vec_env.close()

    row: dict[str, float | int | str] = {
        "algo": str(algo),
        "capacity": int(capacity),
        "train_years": data_years_label(DATA_CONFIG.train_years),
        "eval_years": data_years_label(DATA_CONFIG.eval_years),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    row.update(metrics)
    return row


def main() -> None:
    args = parse_args()
    runner = get_algorithm_runner(args.algo)
    algo_config = runner["config"]
    train_seed = int(algo_config.seed if args.train_seed is None else args.train_seed)
    eval_seed = int(train_seed + 100 if args.eval_seed is None else args.eval_seed)
    total_timesteps = int(algo_config.total_timesteps if args.total_timesteps is None else args.total_timesteps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    plot_dir = experiment_root / "plots"
    experiment_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_historical_data = load_train_historical_data()
    eval_historical_data = load_eval_historical_data()
    split_metadata = get_data_split_metadata(train_historical_data, eval_historical_data)
    results: list[dict[str, float | int | str]] = []
    max_workers = max(1, int(args.max_workers))

    if max_workers == 1:
        for capacity in args.capacities:
            row = run_capacity_job(
                algo=str(args.algo),
                capacity=int(capacity),
                train_historical_data=train_historical_data,
                eval_historical_data=eval_historical_data,
                train_seed=train_seed,
                eval_seed=eval_seed,
                total_timesteps=total_timesteps,
                no_progress_bar=bool(args.no_progress_bar),
                run_prefix=str(args.run_prefix),
            )
            results.append(row)
            print(
                f"[capacity={capacity}] revenue={float(row['episode_revenue']):.2f}, "
                f"reward={float(row['episode_reward']):.2f}, full_day_rate={float(row['full_day_rate']):.4f}"
            )
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_capacity = {
                executor.submit(
                    run_capacity_job,
                    str(args.algo),
                    int(capacity),
                    None,
                    None,
                    train_seed,
                    eval_seed,
                    total_timesteps,
                    bool(args.no_progress_bar),
                    str(args.run_prefix),
                ): int(capacity)
                for capacity in args.capacities
            }
            for future in as_completed(future_to_capacity):
                capacity = future_to_capacity[future]
                row = future.result()
                results.append(row)
                print(
                    f"[capacity={capacity}] revenue={float(row['episode_revenue']):.2f}, "
                    f"reward={float(row['episode_reward']):.2f}, full_day_rate={float(row['full_day_rate']):.4f}"
                )

    results_df = pd.DataFrame(results).sort_values("capacity").reset_index(drop=True)
    csv_path = experiment_root / "capacity_sensitivity_results.csv"
    results_df.to_csv(csv_path, index=False)

    for metric in PLOT_METRICS:
        plot_metric(results_df, metric, plot_dir)

    summary = {
        "capacities": list(map(int, args.capacities)),
        "algo": str(args.algo),
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "total_timesteps": total_timesteps,
        **split_metadata,
        "max_workers": max_workers,
        "results_csv": str(csv_path),
        "plot_dir": str(plot_dir),
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"实验完成，结果表: {csv_path}")
    print(f"单指标图目录: {plot_dir}")


if __name__ == "__main__":
    main()
