from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ABM_CONFIG, ENV_CONFIG, PATH_CONFIG
from src.baseline.pricing_baselines import (
    DEFAULT_PRICE_GRID,
    add_derived_metrics,
    evaluate_manual_policy,
    get_inventory_protection_policy,
    get_static_policy,
    rank_static_candidates,
    search_inventory_protection_best,
    summarize_episode_rows,
)
from src.environment.abm_customer_model import load_filtered_historical_data
from src.training.algorithm_registry import get_algorithm_choices, get_algorithm_runner
from src.training.train_ppo import EpisodeMetricsAggregator


DEFAULT_SCENARIO_FILE = PROJECT_ROOT / "configs" / "scenario_policy_training_scenarios.json"
DEFAULT_EVAL_SEEDS = [142, 143, 144, 145, 146]
DEFAULT_ALGOS = ["sac"]
BASELINE_TOP_K = 20
PLOT_METRICS = [
    "episode_revenue_mean",
    "episode_reward_mean",
    "full_day_rate_mean",
    "avg_price_day0_mean",
    "avg_price_day1_mean",
    "avg_price_day2_mean",
]
RATIO_METRICS = [
    "learned_vs_static_grid_best",
    "learned_vs_inventory_protection_best",
]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_name: str
    capacity: int
    flexible_customer_share: float
    lambda_day_mismatch_flex: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="显式机制场景下的 learned policy 训练与强基线评估")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=DEFAULT_SCENARIO_FILE,
        help="JSON 场景列表文件，每个场景包含 scenario_name/capacity/flexible_customer_share/lambda_day_mismatch_flex",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=DEFAULT_ALGOS,
        choices=get_algorithm_choices(),
        help="每个场景要训练的算法列表",
    )
    parser.add_argument("--train-seed", type=int, default=None, help="训练随机种子，默认使用所选算法配置")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=DEFAULT_EVAL_SEEDS, help="评估随机种子列表")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="每个算法每个场景的训练步数，默认使用所选算法配置",
    )
    parser.add_argument("--price-grid", nargs="+", type=float, default=DEFAULT_PRICE_GRID, help="静态基线价格网格")
    parser.add_argument(
        "--baseline-top-k",
        type=int,
        default=BASELINE_TOP_K,
        help="库存保护基线从静态网格排名前 K 的价格三元组中选基准价",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="scenario_policy_training",
        help="实验输出目录和训练 run 名前缀",
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
        help="并行 job 数；learned-policy 训练按 scenario × algo 粒度并行",
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


def scenario_from_payload(scenario_payload: dict[str, float | int | str]) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_name=str(scenario_payload["scenario_name"]),
        capacity=int(scenario_payload["capacity"]),
        flexible_customer_share=float(scenario_payload["flexible_customer_share"]),
        lambda_day_mismatch_flex=float(scenario_payload["lambda_day_mismatch_flex"]),
    )


def add_episode_metadata(
    rows: list[dict[str, float]],
    *,
    scenario: ScenarioSpec,
    strategy_name: str,
    algo: str,
    policy_type: str,
    eval_seeds: list[int],
    train_seed: int | None = None,
    total_timesteps: int | None = None,
    run_name: str = "",
    run_dir: str = "",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for seed, metrics in zip(eval_seeds, rows):
        row: dict[str, Any] = {
            **scenario_fields(scenario),
            "strategy_name": str(strategy_name),
            "algo": str(algo),
            "policy_type": str(policy_type),
            "eval_seed": int(seed),
            "train_seed": "" if train_seed is None else int(train_seed),
            "total_timesteps": "" if total_timesteps is None else int(total_timesteps),
            "run_name": str(run_name),
            "run_dir": str(run_dir),
        }
        row.update(add_derived_metrics(metrics, scenario.capacity))
        if extra:
            row.update(extra)
        enriched_rows.append(row)
    return enriched_rows


def make_strategy_row(
    scenario: ScenarioSpec,
    strategy_name: str,
    algo: str,
    policy_type: str,
    summary: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **scenario_fields(scenario),
        "strategy_name": str(strategy_name),
        "algo": str(algo),
        "policy_type": str(policy_type),
    }
    row.update(summary)
    if extra:
        row.update(extra)
    return row


def evaluate_learned_policy(
    model,
    train_vec_env,
    build_eval_env_fn,
    historical_data: pd.DataFrame,
    scenario: ScenarioSpec,
    eval_seed: int,
) -> dict[str, float]:
    eval_env = build_eval_env_fn(
        train_vec_env=train_vec_env,
        historical_data=historical_data,
        seed=int(eval_seed),
        capacity=int(scenario.capacity),
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


def plot_strategy_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.8))
    for strategy_name, group in df.groupby("strategy_name"):
        ordered = group.sort_values("scenario_name")
        plt.plot(ordered["scenario_name"], ordered[metric], marker="o", linewidth=1.8, label=str(strategy_name))
    plt.xlabel("scenario")
    plt.ylabel(metric)
    plt.title(f"{metric} by scenario")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=160)
    plt.close()


def plot_summary_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.8))
    for algo, group in df.groupby("algo"):
        ordered = group.sort_values("scenario_name")
        plt.plot(ordered["scenario_name"], ordered[metric], marker="o", linewidth=1.8, label=str(algo))
    plt.xlabel("scenario")
    plt.ylabel(metric)
    plt.title(f"{metric} by scenario")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=160)
    plt.close()


def run_baseline_job(
    scenario_payload: dict[str, float | int | str],
    eval_seeds: list[int],
    price_grid: list[float],
    baseline_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scenario = scenario_from_payload(scenario_payload)
    historical_data = load_filtered_historical_data()
    eval_seeds = list(map(int, eval_seeds))

    with apply_abm_scenario(scenario):
        ranked_static = rank_static_candidates(
            historical_data=historical_data,
            capacity=int(scenario.capacity),
            eval_seeds=eval_seeds,
            price_grid=list(map(float, price_grid)),
        )
        _, best_static_prices, static_summary = ranked_static[0]
        static_policy = get_static_policy(best_static_prices)
        static_rows = [
            evaluate_manual_policy(static_policy, historical_data, int(scenario.capacity), seed)
            for seed in eval_seeds
        ]
        static_meta = {"best_static_prices": json.dumps(list(map(float, best_static_prices)), ensure_ascii=False)}

        candidate_count = max(1, min(int(baseline_top_k), len(ranked_static)))
        candidate_prices = [prices for _, prices, _ in ranked_static[:candidate_count]]
        inventory_summary, inventory_meta = search_inventory_protection_best(
            historical_data=historical_data,
            capacity=int(scenario.capacity),
            eval_seeds=eval_seeds,
            candidate_prices=candidate_prices,
        )
        inventory_base_prices = tuple(json.loads(str(inventory_meta["inventory_base_prices"])))
        inventory_scarcity_alpha = float(inventory_meta["inventory_scarcity_alpha"])
        inventory_policy = get_inventory_protection_policy(
            base_prices=inventory_base_prices,
            scarcity_alpha=inventory_scarcity_alpha,
            capacity=int(scenario.capacity),
        )
        inventory_rows = [
            evaluate_manual_policy(inventory_policy, historical_data, int(scenario.capacity), seed)
            for seed in eval_seeds
        ]

        episode_rows: list[dict[str, Any]] = []
        strategy_rows: list[dict[str, Any]] = [
            make_strategy_row(
                scenario,
                strategy_name="static_grid_best",
                algo="baseline",
                policy_type="baseline_static",
                summary=static_summary,
                extra={**static_meta, "baseline_top_k": int(candidate_count)},
            ),
            make_strategy_row(
                scenario,
                strategy_name="inventory_protection_best",
                algo="baseline",
                policy_type="baseline_inventory",
                summary=inventory_summary,
                extra={**inventory_meta, "baseline_top_k": int(candidate_count)},
            ),
        ]
        episode_rows.extend(
            add_episode_metadata(
                static_rows,
                scenario=scenario,
                strategy_name="static_grid_best",
                algo="baseline",
                policy_type="baseline_static",
                eval_seeds=eval_seeds,
                extra={**static_meta, "baseline_top_k": int(candidate_count)},
            )
        )
        episode_rows.extend(
            add_episode_metadata(
                inventory_rows,
                scenario=scenario,
                strategy_name="inventory_protection_best",
                algo="baseline",
                policy_type="baseline_inventory",
                eval_seeds=eval_seeds,
                extra={**inventory_meta, "baseline_top_k": int(candidate_count)},
            )
        )

        baseline_summary = {
            **scenario_fields(scenario),
            "static_grid_best_revenue": float(static_summary["episode_revenue_mean"]),
            "inventory_protection_best_revenue": float(inventory_summary["episode_revenue_mean"]),
            "best_static_prices": str(static_meta["best_static_prices"]),
            "inventory_base_prices": str(inventory_meta["inventory_base_prices"]),
            "inventory_scarcity_alpha": float(inventory_meta["inventory_scarcity_alpha"]),
            "baseline_top_k": int(candidate_count),
        }

    return episode_rows, strategy_rows, baseline_summary


def run_learned_job(
    scenario_payload: dict[str, float | int | str],
    algo: str,
    train_seed: int | None,
    eval_seeds: list[int],
    total_timesteps: int | None,
    no_progress_bar: bool,
    run_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scenario = scenario_from_payload(scenario_payload)
    historical_data = load_filtered_historical_data()
    eval_seeds = list(map(int, eval_seeds))

    with apply_abm_scenario(scenario):
        runner = get_algorithm_runner(algo)
        algo_config = runner["config"]
        train_single_run_fn = runner["train_single_run"]
        build_eval_env_fn = runner["build_eval_env"]
        effective_train_seed = int(algo_config.seed if train_seed is None else train_seed)
        effective_total_timesteps = int(
            algo_config.total_timesteps if total_timesteps is None else total_timesteps
        )
        run_name = f"{run_prefix}_{scenario.scenario_name}_{algo}"
        model, train_vec_env, run_dir = train_single_run_fn(
            run_name=run_name,
            historical_data=historical_data,
            capacity=int(scenario.capacity),
            train_seed=effective_train_seed,
            total_timesteps=effective_total_timesteps,
            progress_bar=not bool(no_progress_bar),
            verbose=1,
        )
        learned_rows = [
            evaluate_learned_policy(
                model=model,
                train_vec_env=train_vec_env,
                build_eval_env_fn=build_eval_env_fn,
                historical_data=historical_data,
                scenario=scenario,
                eval_seed=seed,
            )
            for seed in eval_seeds
        ]
        learned_summary = summarize_episode_rows(learned_rows, int(scenario.capacity))
        train_vec_env.close()

    strategy_row = make_strategy_row(
        scenario,
        strategy_name=str(algo),
        algo=str(algo),
        policy_type="learned",
        summary=learned_summary,
        extra={
            "train_seed": effective_train_seed,
            "eval_seed_count": int(len(eval_seeds)),
            "total_timesteps": effective_total_timesteps,
            "run_name": run_name,
            "run_dir": str(run_dir),
        },
    )
    episode_rows = add_episode_metadata(
        learned_rows,
        scenario=scenario,
        strategy_name=str(algo),
        algo=str(algo),
        policy_type="learned",
        eval_seeds=eval_seeds,
        train_seed=effective_train_seed,
        total_timesteps=effective_total_timesteps,
        run_name=run_name,
        run_dir=str(run_dir),
    )
    learned_summary_row = {
        **scenario_fields(scenario),
        "algo": str(algo),
        "learned_revenue": float(learned_summary["episode_revenue_mean"]),
        "train_seed": effective_train_seed,
        "eval_seed_count": int(len(eval_seeds)),
        "total_timesteps": effective_total_timesteps,
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    return episode_rows, strategy_row, learned_summary_row


def build_scenario_summary_row(learned_row: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, Any]:
    learned_revenue = float(learned_row["learned_revenue"])
    static_revenue = float(baseline_row["static_grid_best_revenue"])
    inventory_revenue = float(baseline_row["inventory_protection_best_revenue"])
    return {
        **scenario_fields(scenario_from_payload(learned_row)),
        "algo": str(learned_row["algo"]),
        "learned_revenue": learned_revenue,
        "static_grid_best_revenue": static_revenue,
        "inventory_protection_best_revenue": inventory_revenue,
        "learned_vs_static_grid_best": float(learned_revenue / max(1e-8, static_revenue)),
        "learned_vs_inventory_protection_best": float(learned_revenue / max(1e-8, inventory_revenue)),
        "best_static_prices": str(baseline_row["best_static_prices"]),
        "inventory_base_prices": str(baseline_row["inventory_base_prices"]),
        "inventory_scarcity_alpha": float(baseline_row["inventory_scarcity_alpha"]),
        "baseline_top_k": int(baseline_row["baseline_top_k"]),
        "train_seed": int(learned_row["train_seed"]),
        "eval_seed_count": int(learned_row["eval_seed_count"]),
        "total_timesteps": int(learned_row["total_timesteps"]),
        "run_name": str(learned_row["run_name"]),
        "run_dir": str(learned_row["run_dir"]),
    }


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenario_file)
    algos = [str(algo) for algo in args.algos]
    eval_seeds = list(map(int, args.eval_seeds))
    price_grid = list(map(float, args.price_grid))
    max_workers = max(1, int(args.max_workers))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    plot_dir = experiment_root / "plots"
    experiment_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict[str, Any]] = []
    strategy_results: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    scenario_payloads = [scenario_fields(scenario) for scenario in scenarios]
    baseline_by_scenario: dict[str, dict[str, Any]] = {}

    if max_workers == 1:
        for scenario_payload in scenario_payloads:
            episode_rows, strategy_rows, baseline_row = run_baseline_job(
                scenario_payload=scenario_payload,
                eval_seeds=eval_seeds,
                price_grid=price_grid,
                baseline_top_k=int(args.baseline_top_k),
            )
            baseline_by_scenario[str(scenario_payload["scenario_name"])] = baseline_row
            episode_results.extend(episode_rows)
            strategy_results.extend(strategy_rows)

        for scenario_payload in scenario_payloads:
            for algo in algos:
                episode_rows, strategy_row, learned_row = run_learned_job(
                    scenario_payload=scenario_payload,
                    algo=str(algo),
                    train_seed=args.train_seed,
                    eval_seeds=eval_seeds,
                    total_timesteps=args.total_timesteps,
                    no_progress_bar=bool(args.no_progress_bar),
                    run_prefix=str(args.run_prefix),
                )
                episode_results.extend(episode_rows)
                strategy_results.append(strategy_row)
                summary = build_scenario_summary_row(
                    learned_row,
                    baseline_by_scenario[str(scenario_payload["scenario_name"])],
                )
                scenario_summaries.append(summary)
                print(
                    f"[{summary['scenario_name']} algo={summary['algo']}] "
                    f"learned={float(summary['learned_revenue']):.2f}, "
                    f"static={float(summary['static_grid_best_revenue']):.2f}, "
                    f"inventory={float(summary['inventory_protection_best_revenue']):.2f}"
                )
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_baseline = {
                executor.submit(
                    run_baseline_job,
                    scenario_payload,
                    eval_seeds,
                    price_grid,
                    int(args.baseline_top_k),
                ): str(scenario_payload["scenario_name"])
                for scenario_payload in scenario_payloads
            }
            for future in as_completed(future_to_baseline):
                scenario_name = future_to_baseline[future]
                episode_rows, strategy_rows, baseline_row = future.result()
                baseline_by_scenario[scenario_name] = baseline_row
                episode_results.extend(episode_rows)
                strategy_results.extend(strategy_rows)
                print(
                    f"[{scenario_name} baseline] "
                    f"static={float(baseline_row['static_grid_best_revenue']):.2f}, "
                    f"inventory={float(baseline_row['inventory_protection_best_revenue']):.2f}"
                )

        learned_jobs = [(scenario_payload, algo) for scenario_payload in scenario_payloads for algo in algos]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            future_to_job = {
                executor.submit(
                    run_learned_job,
                    scenario_payload,
                    str(algo),
                    args.train_seed,
                    eval_seeds,
                    args.total_timesteps,
                    bool(args.no_progress_bar),
                    str(args.run_prefix),
                ): (str(scenario_payload["scenario_name"]), str(algo))
                for scenario_payload, algo in learned_jobs
            }
            for future in as_completed(future_to_job):
                scenario_name, algo = future_to_job[future]
                episode_rows, strategy_row, learned_row = future.result()
                episode_results.extend(episode_rows)
                strategy_results.append(strategy_row)
                summary = build_scenario_summary_row(learned_row, baseline_by_scenario[scenario_name])
                scenario_summaries.append(summary)
                print(
                    f"[{scenario_name} algo={algo}] "
                    f"learned={float(summary['learned_revenue']):.2f}, "
                    f"static={float(summary['static_grid_best_revenue']):.2f}, "
                    f"inventory={float(summary['inventory_protection_best_revenue']):.2f}"
                )

    episode_df = (
        pd.DataFrame(episode_results)
        .sort_values(["scenario_name", "strategy_name", "eval_seed"])
        .reset_index(drop=True)
    )
    strategy_df = (
        pd.DataFrame(strategy_results)
        .sort_values(["scenario_name", "strategy_name"])
        .reset_index(drop=True)
    )
    summary_df = (
        pd.DataFrame(scenario_summaries)
        .sort_values(["scenario_name", "algo"])
        .reset_index(drop=True)
    )

    episode_csv = experiment_root / "eval_episode_results.csv"
    strategy_csv = experiment_root / "strategy_results.csv"
    summary_csv = experiment_root / "scenario_summary.csv"
    episode_df.to_csv(episode_csv, index=False)
    strategy_df.to_csv(strategy_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    for metric in PLOT_METRICS:
        if metric in strategy_df.columns:
            plot_strategy_metric(strategy_df, metric, plot_dir)
    for metric in RATIO_METRICS:
        if metric in summary_df.columns:
            plot_summary_metric(summary_df, metric, plot_dir)

    experiment_summary = {
        "scenarios": scenario_payloads,
        "algos": algos,
        "train_seed_override": args.train_seed,
        "eval_seeds": eval_seeds,
        "total_timesteps_override": args.total_timesteps,
        "price_grid": price_grid,
        "baseline_top_k": int(args.baseline_top_k),
        "max_workers": max_workers,
        "baseline_parallel_unit": "scenario",
        "learned_parallel_unit": "scenario_x_algo",
        "episode_results_csv": str(episode_csv),
        "strategy_results_csv": str(strategy_csv),
        "scenario_summary_csv": str(summary_csv),
        "plot_dir": str(plot_dir),
        "baseline_methods": ["static_grid_best", "inventory_protection_best"],
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(experiment_summary, file, ensure_ascii=False, indent=2)

    print(f"实验完成，逐 seed 测试结果: {episode_csv}")
    print(f"策略汇总表: {strategy_csv}")
    print(f"场景对比表: {summary_csv}")
    print(f"图表目录: {plot_dir}")


if __name__ == "__main__":
    main()
