from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DATA_CONFIG, ENV_CONFIG, PATH_CONFIG
from src.environment.abm_customer_model import load_eval_historical_data, load_train_historical_data
from src.utils.preprocess_data import data_years_label, get_data_split_metadata
from src.training.train_ppo import EpisodeMetricsAggregator
from src.training.algorithm_registry import get_algorithm_choices, get_algorithm_runner


DEFAULT_CAPACITIES = [20, 30, 40, 50, 60]
DEFAULT_MODES = ["with_penalty", "no_penalty"]
PLOT_METRICS = [
    "episode_revenue",
    "episode_reward",
    "episode_penalty",
    "penalty_revenue_ratio",
    "revenue_per_arrival",
    "revenue_per_capacity_day",
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
    parser = argparse.ArgumentParser(description="有无 penalty 的最小对照实验")
    parser.add_argument("--algo", type=str, default="ppo_tanh_gaussian", choices=get_algorithm_choices(), help="训练算法")
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES, help="要扫描的容量列表")
    parser.add_argument("--modes", nargs="+", type=str, default=DEFAULT_MODES, help="对照模式列表")
    parser.add_argument("--train-seed", type=int, default=None, help="训练用随机种子，默认使用所选算法配置")
    parser.add_argument("--eval-seed", type=int, default=None, help="评估用新随机种子，默认使用 train_seed + 100")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="每个容量-模式组合的训练步数，默认使用所选算法配置",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="penalty_ablation",
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


def build_env_overrides(mode: str) -> dict[str, float | int | str]:
    if mode == "with_penalty":
        return {
            "full_capacity_penalty": float(ENV_CONFIG.full_capacity_penalty),
            "penalty_scale_mode": str(ENV_CONFIG.penalty_scale_mode),
            "penalty_capacity_ref": int(ENV_CONFIG.penalty_capacity_ref),
        }
    if mode == "no_penalty":
        return {
            "full_capacity_penalty": 0.0,
            "penalty_scale_mode": "fixed",
            "penalty_capacity_ref": int(ENV_CONFIG.penalty_capacity_ref),
        }
    raise ValueError(f"未知 penalty mode: {mode}")


@contextmanager
def apply_penalty_config(mode: str):
    original_full_penalty = float(ENV_CONFIG.full_capacity_penalty)
    original_scale_mode = str(ENV_CONFIG.penalty_scale_mode)
    original_capacity_ref = int(ENV_CONFIG.penalty_capacity_ref)
    original_scarcity_coef = float(ENV_CONFIG.scarcity_penalty_coef)
    try:
        if mode == "with_penalty":
            ENV_CONFIG.full_capacity_penalty = original_full_penalty
            ENV_CONFIG.penalty_scale_mode = original_scale_mode
            ENV_CONFIG.penalty_capacity_ref = original_capacity_ref
            ENV_CONFIG.scarcity_penalty_coef = original_scarcity_coef
        elif mode == "no_penalty":
            ENV_CONFIG.full_capacity_penalty = 0.0
            ENV_CONFIG.penalty_scale_mode = "fixed"
            ENV_CONFIG.penalty_capacity_ref = original_capacity_ref
            ENV_CONFIG.scarcity_penalty_coef = 0.0
        else:
            raise ValueError(f"未知 penalty mode: {mode}")
        yield
    finally:
        ENV_CONFIG.full_capacity_penalty = original_full_penalty
        ENV_CONFIG.penalty_scale_mode = original_scale_mode
        ENV_CONFIG.penalty_capacity_ref = original_capacity_ref
        ENV_CONFIG.scarcity_penalty_coef = original_scarcity_coef


def evaluate_policy(
    model,
    train_vec_env,
    build_eval_env_fn,
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seed: int,
    env_overrides: dict[str, float | int | str],
) -> dict[str, float]:
    eval_env = build_eval_env_fn(
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
    for mode, group in df.groupby("penalty_mode"):
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


def plot_full_rate_vs_revenue(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for mode, group in df.groupby("penalty_mode"):
        plt.scatter(group["full_day_rate"], group["revenue_per_capacity_day"], s=55, alpha=0.85, label=mode)
        for _, row in group.iterrows():
            plt.annotate(f"cap={int(row['capacity'])}", (row["full_day_rate"], row["revenue_per_capacity_day"]))
    plt.xlabel("full_day_rate")
    plt.ylabel("revenue_per_capacity_day")
    plt.title("full_day_rate vs revenue_per_capacity_day")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "full_day_rate_vs_revenue_per_capacity_day.png", dpi=160)
    plt.close()


def run_ablation_job(
    algo: str,
    capacity: int,
    mode: str,
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

    env_overrides = build_env_overrides(mode)
    run_name = f"{run_prefix}_{algo}_{mode}_cap{capacity}"

    with apply_penalty_config(mode):
        model, train_vec_env, run_dir = train_single_run_fn(
            run_name=run_name,
            historical_data=train_historical_data,
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
            build_eval_env_fn=build_eval_env_fn,
            historical_data=eval_historical_data,
            capacity=int(capacity),
            eval_seed=int(eval_seed),
            env_overrides=env_overrides,
        )
        train_vec_env.close()

    episode_revenue = float(metrics["episode_revenue"])
    episode_penalty = float(metrics["episode_penalty"])
    episode_arrivals = float(metrics["episode_arrivals"])

    row: dict[str, float | int | str] = {
        "algo": str(algo),
        "penalty_mode": str(mode),
        "capacity": int(capacity),
        "train_years": data_years_label(DATA_CONFIG.train_years),
        "eval_years": data_years_label(DATA_CONFIG.eval_years),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "full_capacity_penalty": float(env_overrides["full_capacity_penalty"]),
        "scarcity_penalty_coef": 0.0 if mode == "no_penalty" else float(ENV_CONFIG.scarcity_penalty_coef),
    }
    row.update(metrics)
    row["penalty_revenue_ratio"] = float(episode_penalty / max(1e-8, episode_revenue))
    row["revenue_per_arrival"] = float(episode_revenue / max(1.0, episode_arrivals))
    row["revenue_per_capacity_day"] = float(episode_revenue / max(1.0, capacity * ENV_CONFIG.episode_days))
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
    modes = list(args.modes)
    capacities = list(map(int, args.capacities))
    jobs = [(mode, capacity) for mode in modes for capacity in capacities]

    if max_workers == 1:
        for mode, capacity in jobs:
            row = run_ablation_job(
                algo=str(args.algo),
                capacity=int(capacity),
                mode=str(mode),
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
                f"[mode={mode}, capacity={capacity}] revenue={float(row['episode_revenue']):.2f}, "
                f"full_day_rate={float(row['full_day_rate']):.4f}, "
                f"penalty_ratio={float(row['penalty_revenue_ratio']):.6f}"
            )
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_job = {
                executor.submit(
                    run_ablation_job,
                    str(args.algo),
                    int(capacity),
                    str(mode),
                    None,
                    None,
                    train_seed,
                    eval_seed,
                    total_timesteps,
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
                    f"full_day_rate={float(row['full_day_rate']):.4f}, "
                    f"penalty_ratio={float(row['penalty_revenue_ratio']):.6f}"
                )

    results_df = pd.DataFrame(results).sort_values(["penalty_mode", "capacity"]).reset_index(drop=True)
    csv_path = experiment_root / "penalty_ablation_results.csv"
    results_df.to_csv(csv_path, index=False)

    for metric in PLOT_METRICS:
        plot_metric(results_df, metric, plot_dir)
    plot_full_rate_vs_revenue(results_df, plot_dir)

    summary = {
        "modes": modes,
        "capacities": capacities,
        "algo": str(args.algo),
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "total_timesteps": total_timesteps,
        "base_full_capacity_penalty": float(ENV_CONFIG.full_capacity_penalty),
        "base_scarcity_penalty_coef": float(ENV_CONFIG.scarcity_penalty_coef),
        "base_penalty_scale_mode": str(ENV_CONFIG.penalty_scale_mode),
        "episode_days": int(ENV_CONFIG.episode_days),
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
