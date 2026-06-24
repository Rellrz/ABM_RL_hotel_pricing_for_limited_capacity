from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PATH_CONFIG, PPO_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data
from src.training.train_ppo import EpisodeMetricsAggregator, build_eval_env, train_single_run


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
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES, help="要扫描的容量列表")
    parser.add_argument("--train-seed", type=int, default=int(PPO_CONFIG.seed), help="训练用随机种子")
    parser.add_argument("--eval-seed", type=int, default=int(PPO_CONFIG.seed) + 100, help="评估用新随机种子")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(PPO_CONFIG.total_timesteps),
        help="每个容量的训练步数",
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
    return parser.parse_args()


def evaluate_policy(model, train_vec_env, historical_data: pd.DataFrame, capacity: int, eval_seed: int) -> dict[str, float]:
    eval_env = build_eval_env(
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


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    plot_dir = experiment_root / "plots"
    experiment_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    historical_data = load_filtered_historical_data()
    results: list[dict[str, float | int | str]] = []

    for capacity in args.capacities:
        run_name = f"{args.run_prefix}_cap{capacity}"
        model, train_vec_env, run_dir = train_single_run(
            run_name=run_name,
            historical_data=historical_data,
            capacity=int(capacity),
            train_seed=int(args.train_seed),
            total_timesteps=int(args.total_timesteps),
            progress_bar=not bool(args.no_progress_bar),
            verbose=1,
        )
        metrics = evaluate_policy(
            model=model,
            train_vec_env=train_vec_env,
            historical_data=historical_data,
            capacity=int(capacity),
            eval_seed=int(args.eval_seed),
        )
        train_vec_env.close()

        row: dict[str, float | int | str] = {
            "capacity": int(capacity),
            "train_seed": int(args.train_seed),
            "eval_seed": int(args.eval_seed),
            "run_name": run_name,
            "run_dir": str(run_dir),
        }
        row.update(metrics)
        results.append(row)
        print(
            f"[capacity={capacity}] revenue={metrics['episode_revenue']:.2f}, "
            f"reward={metrics['episode_reward']:.2f}, full_day_rate={metrics['full_day_rate']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("capacity").reset_index(drop=True)
    csv_path = experiment_root / "capacity_sensitivity_results.csv"
    results_df.to_csv(csv_path, index=False)

    for metric in PLOT_METRICS:
        plot_metric(results_df, metric, plot_dir)

    summary = {
        "capacities": list(map(int, args.capacities)),
        "train_seed": int(args.train_seed),
        "eval_seed": int(args.eval_seed),
        "total_timesteps": int(args.total_timesteps),
        "results_csv": str(csv_path),
        "plot_dir": str(plot_dir),
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"实验完成，结果表: {csv_path}")
    print(f"单指标图目录: {plot_dir}")


if __name__ == "__main__":
    main()
