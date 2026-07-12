from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ABM_CONFIG, ENV_CONFIG, PATH_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data
from src.training.train_ppo import EpisodeMetricsAggregator
from src.training.algorithm_registry import get_algorithm_choices, get_algorithm_runner


DEFAULT_SCENARIO_FILE = PROJECT_ROOT / "configs" / "scenario_policy_training_scenarios.json"
DEFAULT_MODES = ["scarcity_0", "scarcity_3000", "scarcity_9000"]
DEFAULT_EVAL_SEEDS = [142, 143, 144]
PLOT_METRICS = [
    "episode_revenue_mean",
    "episode_reward_mean",
    "episode_penalty_mean",
    "penalty_revenue_ratio_mean",
    "revenue_per_arrival_mean",
    "revenue_per_capacity_day_mean",
    "episode_acceptance_rate_mean",
    "full_day_rate_mean",
    "full_rate_day0_mean",
    "full_rate_day1_mean",
    "full_rate_day2_mean",
    "avg_price_day0_mean",
    "avg_price_day1_mean",
    "avg_price_day2_mean",
]
SUMMARY_METRICS = [
    "episode_revenue",
    "episode_raw_reward",
    "episode_reward",
    "episode_penalty",
    "episode_scarcity_penalty",
    "episode_arrivals",
    "episode_accepted",
    "episode_rejected",
    "episode_acceptance_rate",
    "avg_price",
    "avg_inventory",
    "avg_inventory_before",
    "full_day_rate",
    "full_slot_rate",
    "avg_price_day0",
    "avg_price_day1",
    "avg_price_day2",
    "avg_inventory_day0",
    "avg_inventory_day1",
    "avg_inventory_day2",
    "avg_inventory_before_day0",
    "avg_inventory_before_day1",
    "avg_inventory_before_day2",
    "full_rate_day0",
    "full_rate_day1",
    "full_rate_day2",
    "avg_rejected_day0",
    "avg_rejected_day1",
    "avg_rejected_day2",
    "penalty_revenue_ratio",
    "revenue_per_arrival",
    "revenue_per_capacity_day",
]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_name: str
    capacity: int
    flexible_customer_share: float
    lambda_day_mismatch_flex: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scarcity penalty 系数消融实验")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=DEFAULT_SCENARIO_FILE,
        help="JSON 场景列表文件，每个场景包含 scenario_name/capacity/flexible_customer_share/lambda_day_mismatch_flex",
    )
    parser.add_argument("--algo", type=str, default="ppo_beta", choices=get_algorithm_choices(), help="训练算法")
    parser.add_argument(
        "--modes",
        nargs="+",
        type=str,
        default=DEFAULT_MODES,
        help=(
            "对照模式列表。支持 scarcity_<coef>，例如 scarcity_0、scarcity_3000、scarcity_9000。"
        ),
    )
    parser.add_argument("--train-seed", type=int, default=None, help="训练用随机种子，默认使用所选算法配置")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=DEFAULT_EVAL_SEEDS, help="评估随机种子列表")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="每个场景-模式组合的训练步数，默认使用所选算法配置",
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
        help="并行进程数，设为大于 1 时会对不同场景-模式组合并行训练与评估",
    )
    return parser.parse_args()


def load_scenarios(path: Path) -> list[ScenarioSpec]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("scenario-file 必须是 JSON list。")

    scenarios: list[ScenarioSpec] = []
    seen_names: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("每个 scenario 必须是 JSON object。")
        scenario = ScenarioSpec(
            scenario_name=str(raw["scenario_name"]).strip(),
            capacity=int(raw["capacity"]),
            flexible_customer_share=float(raw["flexible_customer_share"]),
            lambda_day_mismatch_flex=float(raw["lambda_day_mismatch_flex"]),
        )
        if not scenario.scenario_name:
            raise ValueError("scenario_name 不能为空。")
        if scenario.scenario_name in seen_names:
            raise ValueError(f"重复 scenario_name: {scenario.scenario_name}")
        if scenario.capacity <= 0:
            raise ValueError(f"{scenario.scenario_name}: capacity 必须为正数。")
        if not (0.0 <= scenario.flexible_customer_share <= 1.0):
            raise ValueError(f"{scenario.scenario_name}: flexible_customer_share 必须位于 [0, 1]。")
        if scenario.lambda_day_mismatch_flex < 0.0:
            raise ValueError(f"{scenario.scenario_name}: lambda_day_mismatch_flex 不能为负数。")
        seen_names.add(scenario.scenario_name)
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError("scenario-file 至少需要包含一个场景。")
    return scenarios


@contextmanager
def apply_abm_scenario(scenario: ScenarioSpec):
    original_flexible_customer_share = float(ABM_CONFIG.flexible_customer_share)
    original_lambda_day_mismatch_flex = float(ABM_CONFIG.lambda_day_mismatch_flex)
    try:
        ABM_CONFIG.flexible_customer_share = float(scenario.flexible_customer_share)
        ABM_CONFIG.lambda_day_mismatch_flex = float(scenario.lambda_day_mismatch_flex)
        yield
    finally:
        ABM_CONFIG.flexible_customer_share = original_flexible_customer_share
        ABM_CONFIG.lambda_day_mismatch_flex = original_lambda_day_mismatch_flex


def scenario_fields(scenario: ScenarioSpec) -> dict[str, float | int | str]:
    return {
        "scenario_name": str(scenario.scenario_name),
        "capacity": int(scenario.capacity),
        "flexible_customer_share": float(scenario.flexible_customer_share),
        "lambda_day_mismatch_flex": float(scenario.lambda_day_mismatch_flex),
    }


def _parse_scarcity_coef(mode: str) -> float:
    if mode == "scarcity":
        return float(ENV_CONFIG.scarcity_penalty_coef)
    prefix = "scarcity_"
    if mode.startswith(prefix):
        return float(mode.removeprefix(prefix))
    raise ValueError(f"未知 scarcity mode: {mode}")


def build_env_overrides(mode: str) -> dict[str, Any]:
    coef = _parse_scarcity_coef(mode)
    if coef < 0.0:
        raise ValueError("scarcity penalty 系数不能为负数。")
    return {
        "scarcity_threshold_ratio": float(ENV_CONFIG.scarcity_threshold_ratio),
        "scarcity_penalty_coef": float(coef),
        "scarcity_penalty_weights": [0.0, 0.5, 1.0],
    }


def _format_env_overrides(env_overrides: dict[str, Any]) -> str:
    return json.dumps(env_overrides, ensure_ascii=False, sort_keys=True)


def add_derived_metrics(metrics: dict[str, float], capacity: int) -> dict[str, float]:
    enriched = dict(metrics)
    episode_revenue = float(enriched["episode_revenue"])
    episode_penalty = float(enriched["episode_penalty"])
    episode_arrivals = float(enriched["episode_arrivals"])
    enriched["penalty_revenue_ratio"] = float(episode_penalty / max(1e-8, episode_revenue))
    enriched["revenue_per_arrival"] = float(episode_revenue / max(1.0, episode_arrivals))
    enriched["revenue_per_capacity_day"] = float(
        episode_revenue / max(1.0, float(capacity) * float(ENV_CONFIG.episode_days))
    )
    return enriched


def summarize_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for metric in SUMMARY_METRICS:
        summary[f"{metric}_mean"] = float(frame[metric].mean())
        summary[f"{metric}_std"] = float(frame[metric].std(ddof=0))
    return summary


def validate_modes(modes: list[str]) -> None:
    for mode in modes:
        build_env_overrides(str(mode))


def evaluate_policy(
    model,
    train_vec_env,
    build_eval_env_fn,
    historical_data: pd.DataFrame,
    capacity: int,
    eval_seed: int,
    env_overrides: dict[str, Any],
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

    plt.figure(figsize=(8, 4))
    for mode, group in df.groupby("penalty_mode"):
        ordered = group.sort_values("scenario_name")
        plt.plot(ordered["scenario_name"], ordered[metric], marker="o", linewidth=1.8, label=mode)
    plt.xlabel("scenario")
    plt.ylabel(metric)
    plt.title(f"{metric} vs scenario")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=160)
    plt.close()


def plot_full_rate_vs_revenue(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for mode, group in df.groupby("penalty_mode"):
        plt.scatter(
            group["full_day_rate_mean"],
            group["revenue_per_capacity_day_mean"],
            s=55,
            alpha=0.85,
            label=mode,
        )
        for _, row in group.iterrows():
            label = str(row["scenario_name"]).replace("scenario_", "")
            plt.annotate(label, (row["full_day_rate_mean"], row["revenue_per_capacity_day_mean"]))
    plt.xlabel("full_day_rate_mean")
    plt.ylabel("revenue_per_capacity_day_mean")
    plt.title("full_day_rate_mean vs revenue_per_capacity_day_mean")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "full_day_rate_vs_revenue_per_capacity_day.png", dpi=160)
    plt.close()


def run_ablation_job(
    algo: str,
    scenario: ScenarioSpec,
    mode: str,
    historical_data: pd.DataFrame | None,
    train_seed: int,
    eval_seeds: list[int],
    total_timesteps: int,
    no_progress_bar: bool,
    run_prefix: str,
) -> tuple[dict[str, float | int | str], list[dict[str, Any]]]:
    if historical_data is None:
        historical_data = load_filtered_historical_data()
    runner = get_algorithm_runner(algo)
    train_single_run_fn = runner["train_single_run"]
    build_eval_env_fn = runner["build_eval_env"]

    env_overrides = build_env_overrides(mode)
    run_name = f"{run_prefix}_{scenario.scenario_name}_{algo}_{mode}"

    with apply_abm_scenario(scenario):
        model, train_vec_env, run_dir = train_single_run_fn(
            run_name=run_name,
            historical_data=historical_data,
            capacity=int(scenario.capacity),
            train_seed=int(train_seed),
            total_timesteps=int(total_timesteps),
            progress_bar=not bool(no_progress_bar),
            verbose=1,
            env_overrides=env_overrides,
        )
        eval_rows: list[dict[str, Any]] = []
        for eval_seed in eval_seeds:
            metrics = evaluate_policy(
                model=model,
                train_vec_env=train_vec_env,
                build_eval_env_fn=build_eval_env_fn,
                historical_data=historical_data,
                capacity=int(scenario.capacity),
                eval_seed=int(eval_seed),
                env_overrides=env_overrides,
            )
            enriched_metrics = add_derived_metrics(metrics, int(scenario.capacity))
            eval_rows.append(
                {
                    **scenario_fields(scenario),
                    "algo": str(algo),
                    "penalty_mode": str(mode),
                    "train_seed": int(train_seed),
                    "eval_seed": int(eval_seed),
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "env_overrides": _format_env_overrides(env_overrides),
                    "scarcity_threshold_ratio": float(env_overrides["scarcity_threshold_ratio"]),
                    "scarcity_penalty_coef": float(env_overrides["scarcity_penalty_coef"]),
                    "scarcity_penalty_weights": json.dumps(
                        env_overrides["scarcity_penalty_weights"],
                        ensure_ascii=False,
                    ),
                    **enriched_metrics,
                }
            )
        train_vec_env.close()

    row: dict[str, float | int | str] = {
        **scenario_fields(scenario),
        "algo": str(algo),
        "penalty_mode": str(mode),
        "train_seed": int(train_seed),
        "eval_seed_count": int(len(eval_seeds)),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "env_overrides": _format_env_overrides(env_overrides),
        "scarcity_threshold_ratio": float(env_overrides["scarcity_threshold_ratio"]),
        "scarcity_penalty_coef": float(env_overrides["scarcity_penalty_coef"]),
        "scarcity_penalty_weights": json.dumps(env_overrides["scarcity_penalty_weights"], ensure_ascii=False),
    }
    row.update(summarize_eval_rows(eval_rows))
    return row, eval_rows


def main() -> None:
    args = parse_args()
    runner = get_algorithm_runner(args.algo)
    algo_config = runner["config"]
    train_seed = int(algo_config.seed if args.train_seed is None else args.train_seed)
    eval_seeds = list(map(int, args.eval_seeds))
    total_timesteps = int(algo_config.total_timesteps if args.total_timesteps is None else args.total_timesteps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    plot_dir = experiment_root / "plots"
    experiment_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    historical_data = load_filtered_historical_data()
    results: list[dict[str, float | int | str]] = []
    eval_results: list[dict[str, Any]] = []
    max_workers = max(1, int(args.max_workers))
    modes = list(args.modes)
    validate_modes(modes)
    scenario_file = Path(args.scenario_file)
    scenarios = load_scenarios(scenario_file)
    jobs = [(mode, scenario) for mode in modes for scenario in scenarios]

    if max_workers == 1:
        for mode, scenario in jobs:
            row, eval_rows = run_ablation_job(
                algo=str(args.algo),
                scenario=scenario,
                mode=str(mode),
                historical_data=historical_data,
                train_seed=train_seed,
                eval_seeds=eval_seeds,
                total_timesteps=total_timesteps,
                no_progress_bar=bool(args.no_progress_bar),
                run_prefix=str(args.run_prefix),
            )
            results.append(row)
            eval_results.extend(eval_rows)
            print(
                f"[mode={mode}, scenario={scenario.scenario_name}] "
                f"revenue_mean={float(row['episode_revenue_mean']):.2f}, "
                f"full_day_rate_mean={float(row['full_day_rate_mean']):.4f}, "
                f"penalty_ratio_mean={float(row['penalty_revenue_ratio_mean']):.6f}"
            )
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_job = {
                executor.submit(
                    run_ablation_job,
                    str(args.algo),
                    scenario,
                    str(mode),
                    None,
                    train_seed,
                    eval_seeds,
                    total_timesteps,
                    bool(args.no_progress_bar),
                    str(args.run_prefix),
                ): (mode, scenario)
                for mode, scenario in jobs
            }
            for future in as_completed(future_to_job):
                mode, scenario = future_to_job[future]
                row, eval_rows = future.result()
                results.append(row)
                eval_results.extend(eval_rows)
                print(
                    f"[mode={mode}, scenario={scenario.scenario_name}] "
                    f"revenue_mean={float(row['episode_revenue_mean']):.2f}, "
                    f"full_day_rate_mean={float(row['full_day_rate_mean']):.4f}, "
                    f"penalty_ratio_mean={float(row['penalty_revenue_ratio_mean']):.6f}"
                )

    results_df = pd.DataFrame(results).sort_values(["penalty_mode", "scenario_name"]).reset_index(drop=True)
    eval_results_df = (
        pd.DataFrame(eval_results)
        .sort_values(["penalty_mode", "scenario_name", "train_seed", "eval_seed"])
        .reset_index(drop=True)
    )
    csv_path = experiment_root / "penalty_ablation_results.csv"
    eval_csv_path = experiment_root / "penalty_ablation_eval_results.csv"
    results_df.to_csv(csv_path, index=False)
    eval_results_df.to_csv(eval_csv_path, index=False)

    for metric in PLOT_METRICS:
        plot_metric(results_df, metric, plot_dir)
    plot_full_rate_vs_revenue(results_df, plot_dir)

    summary = {
        "modes": modes,
        "scenario_file": str(scenario_file),
        "scenarios": [scenario_fields(scenario) for scenario in scenarios],
        "algo": str(args.algo),
        "train_seed": train_seed,
        "eval_seeds": eval_seeds,
        "total_timesteps": total_timesteps,
        "base_scarcity_penalty_coef": float(ENV_CONFIG.scarcity_penalty_coef),
        "base_scarcity_penalty_weights": list(map(float, ENV_CONFIG.scarcity_penalty_weights)),
        "base_scarcity_threshold_ratio": float(ENV_CONFIG.scarcity_threshold_ratio),
        "episode_days": int(ENV_CONFIG.episode_days),
        "max_workers": max_workers,
        "results_csv": str(csv_path),
        "eval_results_csv": str(eval_csv_path),
        "plot_dir": str(plot_dir),
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"实验完成，汇总结果表: {csv_path}")
    print(f"逐 eval seed 结果表: {eval_csv_path}")
    print(f"单指标图目录: {plot_dir}")


if __name__ == "__main__":
    main()
