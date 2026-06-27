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

from configs.config import ENV_CONFIG, PATH_CONFIG, PPO_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data
from src.training.train_ppo import EpisodeMetricsAggregator, build_eval_env, train_single_run


DEFAULT_CAPACITIES = [20, 30, 40, 50, 60]
DEFAULT_MODES = ["fixed", "linear_capacity"]
PLOT_METRICS = [
    "effective_full_penalty",
    "episode_revenue",
    "episode_reward",
    "episode_penalty",
    "penalty_revenue_ratio",
    "penalty_per_accepted",
    "episode_acceptance_rate",
    "full_day_rate",
    "full_rate_day0",
    "full_rate_day1",
    "full_rate_day2",
    "avg_price_day0",
    "avg_price_day1",
    "avg_price_day2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="惩罚缩放机制对照实验")
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES, help="要扫描的容量列表")
    parser.add_argument("--modes", nargs="+", type=str, default=DEFAULT_MODES, help="惩罚缩放模式列表")
    parser.add_argument("--train-seed", type=int, default=int(PPO_CONFIG.seed), help="训练用随机种子")
    parser.add_argument("--eval-seed", type=int, default=int(PPO_CONFIG.seed) + 100, help="评估用新随机种子")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(PPO_CONFIG.total_timesteps),
        help="每个容量-模式组合的训练步数",
    )
    parser.add_argument(
        "--base-penalty",
        type=float,
        default=float(ENV_CONFIG.full_capacity_penalty),
        help="full_capacity_penalty 的基准值",
    )
    parser.add_argument(
        "--penalty-capacity-ref",
        type=int,
        default=int(ENV_CONFIG.penalty_capacity_ref),
        help="线性缩放模式下的参考容量",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="penalty_scaling",
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
        help="并行进程数，设为大于 1 时会对不同组合并行训练与评估",
    )
    return parser.parse_args()


def compute_effective_full_penalty(mode: str, capacity: int, base_penalty: float, penalty_capacity_ref: int) -> float:
    if mode == "fixed":
        return float(base_penalty)
    if mode == "linear_capacity":
        return float(base_penalty * capacity / max(1, penalty_capacity_ref))
    raise ValueError(f"未知 penalty scale mode: {mode}")


def evaluate_policy(
    model,
    train_vec_env,
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seed: int,
    env_overrides: dict[str, float | int | str],
) -> dict[str, float]:
    eval_env = build_eval_env(
        train_vec_env=train_vec_env,
        historical_data=historical_data,
        seed=eval_seed,
        capacity=capacity,
        env_overrides=env_overrides,
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
    for mode, group in df.groupby("penalty_scale_mode"):
        ordered = group.sort_values("capacity")
        plt.plot(ordered["capacity"], ordered[metric], marker="o", linewidth=1.8, label=mode)
    plt.xlabel("capacity")
    plt.ylabel(metric)
    plt.title(f"{metric} vs capacity")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=160)
    plt.close()


def run_penalty_job(
    capacity: int,
    mode: str,
    historical_data: pd.DataFrame | None,
    train_seed: int,
    eval_seed: int,
    total_timesteps: int,
    base_penalty: float,
    penalty_capacity_ref: int,
    no_progress_bar: bool,
    run_prefix: str,
) -> dict[str, float | int | str]:
    if historical_data is None:
        historical_data = load_filtered_historical_data()

    env_overrides = {
        "full_capacity_penalty": float(base_penalty),
        "penalty_scale_mode": str(mode),
        "penalty_capacity_ref": int(penalty_capacity_ref),
    }
    run_name = f"{run_prefix}_{mode}_cap{capacity}"
    model, train_vec_env, run_dir = train_single_run(
        run_name=run_name,
        historical_data=historical_data,
        capacity=int(capacity),
        train_seed=int(train_seed),
        total_timesteps=int(total_timesteps),
        progress_bar=not bool(no_progress_bar),
        verbose=1,
        env_overrides=env_overrides,
    )
    metrics = evaluate_policy(
        model=model,
        train_vec_env=train_vec_env,
        historical_data=historical_data,
        capacity=int(capacity),
        eval_seed=int(eval_seed),
        env_overrides=env_overrides,
    )
    train_vec_env.close()

    effective_penalty = compute_effective_full_penalty(
        mode=mode,
        capacity=int(capacity),
        base_penalty=float(base_penalty),
        penalty_capacity_ref=int(penalty_capacity_ref),
    )
    episode_revenue = float(metrics["episode_revenue"])
    episode_penalty = float(metrics["episode_penalty"])
    episode_accepted = float(metrics["episode_accepted"])

    row: dict[str, float | int | str] = {
        "penalty_scale_mode": str(mode),
        "capacity": int(capacity),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "base_penalty": float(base_penalty),
        "penalty_capacity_ref": int(penalty_capacity_ref),
        "effective_full_penalty": float(effective_penalty),
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    row.update(metrics)
    row["penalty_revenue_ratio"] = float(episode_penalty / max(1e-8, episode_revenue))
    row["penalty_per_accepted"] = float(episode_penalty / max(1.0, episode_accepted))
    return row


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    plot_dir = experiment_root / "plots"
    experiment_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    historical_data = load_filtered_historical_data()
    results: list[dict[str, float | int | str]] = []
    max_workers = max(1, int(args.max_workers))
    modes = list(args.modes)
    capacities = list(map(int, args.capacities))
    jobs = [(mode, capacity) for mode in modes for capacity in capacities]

    if max_workers == 1:
        for mode, capacity in jobs:
            row = run_penalty_job(
                capacity=int(capacity),
                mode=str(mode),
                historical_data=historical_data,
                train_seed=int(args.train_seed),
                eval_seed=int(args.eval_seed),
                total_timesteps=int(args.total_timesteps),
                base_penalty=float(args.base_penalty),
                penalty_capacity_ref=int(args.penalty_capacity_ref),
                no_progress_bar=bool(args.no_progress_bar),
                run_prefix=str(args.run_prefix),
            )
            results.append(row)
            print(
                f"[mode={mode}, capacity={capacity}] revenue={float(row['episode_revenue']):.2f}, "
                f"penalty_ratio={float(row['penalty_revenue_ratio']):.6f}"
            )
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_job = {
                executor.submit(
                    run_penalty_job,
                    int(capacity),
                    str(mode),
                    None,
                    int(args.train_seed),
                    int(args.eval_seed),
                    int(args.total_timesteps),
                    float(args.base_penalty),
                    int(args.penalty_capacity_ref),
                    bool(args.no_progress_bar),
                    str(args.run_prefix),
                ): (mode, capacity)
                for mode, capacity in jobs
            }
            for future in as_completed(future_to_job):
                mode, capacity = future_to_job[future]
                row = future.result()
                results.append(row)
                print(
                    f"[mode={mode}, capacity={capacity}] revenue={float(row['episode_revenue']):.2f}, "
                    f"penalty_ratio={float(row['penalty_revenue_ratio']):.6f}"
                )

    results_df = (
        pd.DataFrame(results)
        .sort_values(["penalty_scale_mode", "capacity"])
        .reset_index(drop=True)
    )
    csv_path = experiment_root / "penalty_scaling_results.csv"
    results_df.to_csv(csv_path, index=False)

    for metric in PLOT_METRICS:
        plot_metric(results_df, metric, plot_dir)

    summary = {
        "modes": modes,
        "capacities": capacities,
        "train_seed": int(args.train_seed),
        "eval_seed": int(args.eval_seed),
        "total_timesteps": int(args.total_timesteps),
        "base_penalty": float(args.base_penalty),
        "penalty_capacity_ref": int(args.penalty_capacity_ref),
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
