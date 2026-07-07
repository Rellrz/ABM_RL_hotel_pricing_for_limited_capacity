from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ABM_CONFIG, DATA_CONFIG, PATH_CONFIG, PPO_CONFIG, WARM_START_CONFIG
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
from src.environment.abm_customer_model import load_train_historical_data
from src.training.algorithm_registry import get_algorithm_runner
from src.training.train_ppo import EpisodeMetricsAggregator
from src.training.warm_start import train_ppo_beta_warm_start
from src.utils.preprocess_data import data_years_label, get_data_split_metadata


DEFAULT_SCENARIO_FILE = PROJECT_ROOT / "configs" / "scenario_policy_training_scenarios.json"
DEFAULT_SCENARIO_NAMES = ["scenario_b_cap20_flex075_lam48"]
DEFAULT_SELECTION_SEEDS = [42, 43, 44]
DEFAULT_EVAL_SEEDS = [142, 143, 144]
BASELINE_TOP_K = 20


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_name: str
    capacity: int
    flexible_customer_share: float
    lambda_day_mismatch_flex: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO-beta inventory-protect warm-start 验证实验")
    parser.add_argument("--scenario-file", type=Path, default=DEFAULT_SCENARIO_FILE)
    parser.add_argument(
        "--scenario-names",
        nargs="+",
        default=DEFAULT_SCENARIO_NAMES,
        help="默认只跑 Scenario B；传入 all 可跑全部场景。",
    )
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=DEFAULT_SELECTION_SEEDS)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--demo-episodes", type=int, default=WARM_START_CONFIG.demo_episodes)
    parser.add_argument("--bc-epochs", type=int, default=WARM_START_CONFIG.bc_epochs)
    parser.add_argument("--bc-batch-size", type=int, default=WARM_START_CONFIG.bc_batch_size)
    parser.add_argument("--bc-learning-rate", type=float, default=WARM_START_CONFIG.bc_learning_rate)
    parser.add_argument("--bc-entropy-coef", type=float, default=WARM_START_CONFIG.bc_entropy_coef)
    parser.add_argument("--price-grid", nargs="+", type=float, default=DEFAULT_PRICE_GRID)
    parser.add_argument("--baseline-top-k", type=int, default=BASELINE_TOP_K)
    parser.add_argument("--run-prefix", type=str, default="warm_start_policy")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def load_scenarios(path: Path, scenario_names: list[str]) -> list[ScenarioSpec]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    wanted = {name.strip() for name in scenario_names}
    include_all = wanted == {"all"}
    scenarios: list[ScenarioSpec] = []
    for raw in payload:
        scenario = ScenarioSpec(
            scenario_name=str(raw["scenario_name"]),
            capacity=int(raw["capacity"]),
            flexible_customer_share=float(raw["flexible_customer_share"]),
            lambda_day_mismatch_flex=float(raw["lambda_day_mismatch_flex"]),
        )
        if include_all or scenario.scenario_name in wanted:
            scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"没有匹配的场景: {scenario_names}")
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


def scenario_fields(scenario: ScenarioSpec) -> dict[str, Any]:
    years = data_years_label(DATA_CONFIG.train_years)
    return {
        "scenario_name": scenario.scenario_name,
        "capacity": int(scenario.capacity),
        "flexible_customer_share": float(scenario.flexible_customer_share),
        "lambda_day_mismatch_flex": float(scenario.lambda_day_mismatch_flex),
        "train_years": years,
        "test_years": years,
        "distribution_mode": "same_distribution_train_years",
    }


def evaluate_learned_policy(model, train_vec_env, historical_data, scenario: ScenarioSpec, eval_seed: int):
    build_eval_env_fn = get_algorithm_runner("ppo_beta")["build_eval_env"]
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


def add_episode_rows(
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
    enriched: list[dict[str, Any]] = []
    for seed, metrics in zip(eval_seeds, rows):
        row: dict[str, Any] = {
            **scenario_fields(scenario),
            "strategy_name": strategy_name,
            "algo": algo,
            "policy_type": policy_type,
            "eval_seed": int(seed),
            "train_seed": "" if train_seed is None else int(train_seed),
            "total_timesteps": "" if total_timesteps is None else int(total_timesteps),
            "run_name": run_name,
            "run_dir": run_dir,
        }
        row.update(add_derived_metrics(metrics, int(scenario.capacity)))
        if extra:
            row.update(extra)
        enriched.append(row)
    return enriched


def make_strategy_row(
    *,
    scenario: ScenarioSpec,
    strategy_name: str,
    algo: str,
    policy_type: str,
    summary: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **scenario_fields(scenario),
        "strategy_name": strategy_name,
        "algo": algo,
        "policy_type": policy_type,
    }
    row.update(summary)
    if extra:
        row.update(extra)
    return row


def run_scenario(
    *,
    scenario: ScenarioSpec,
    train_historical_data: pd.DataFrame,
    selection_seeds: list[int],
    eval_seeds: list[int],
    price_grid: list[float],
    baseline_top_k: int,
    train_seed: int | None,
    total_timesteps: int,
    run_prefix: str,
    no_progress_bar: bool,
    demo_episodes: int,
    bc_epochs: int,
    bc_batch_size: int,
    bc_learning_rate: float,
    bc_entropy_coef: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with apply_abm_scenario(scenario):
        ranked_static = rank_static_candidates(
            historical_data=train_historical_data,
            capacity=int(scenario.capacity),
            eval_seeds=selection_seeds,
            price_grid=price_grid,
        )
        _, best_static_prices, _ = ranked_static[0]
        candidate_count = max(1, min(int(baseline_top_k), len(ranked_static)))
        candidate_prices = [prices for _, prices, _ in ranked_static[:candidate_count]]
        _, inventory_meta = search_inventory_protection_best(
            historical_data=train_historical_data,
            capacity=int(scenario.capacity),
            eval_seeds=selection_seeds,
            candidate_prices=candidate_prices,
        )
        inventory_base_prices = tuple(json.loads(str(inventory_meta["inventory_base_prices"])))
        inventory_scarcity_alpha = float(inventory_meta["inventory_scarcity_alpha"])

        static_policy = get_static_policy(best_static_prices)
        inventory_policy = get_inventory_protection_policy(
            base_prices=inventory_base_prices,
            scarcity_alpha=inventory_scarcity_alpha,
            capacity=int(scenario.capacity),
        )
        static_rows = [
            evaluate_manual_policy(static_policy, train_historical_data, int(scenario.capacity), seed)
            for seed in eval_seeds
        ]
        inventory_rows = [
            evaluate_manual_policy(inventory_policy, train_historical_data, int(scenario.capacity), seed)
            for seed in eval_seeds
        ]
        static_summary = summarize_episode_rows(static_rows, int(scenario.capacity))
        inventory_summary = summarize_episode_rows(inventory_rows, int(scenario.capacity))

        effective_train_seed = int(PPO_CONFIG.seed if train_seed is None else train_seed)
        direct_run_name = f"{run_prefix}_{scenario.scenario_name}_ppo_beta_direct"
        direct_runner = get_algorithm_runner("ppo_beta")
        direct_model, direct_vec_env, direct_run_dir = direct_runner["train_single_run"](
            run_name=direct_run_name,
            historical_data=train_historical_data,
            capacity=int(scenario.capacity),
            train_seed=effective_train_seed,
            total_timesteps=int(total_timesteps),
            progress_bar=not bool(no_progress_bar),
            verbose=1,
        )
        direct_rows = [
            evaluate_learned_policy(direct_model, direct_vec_env, train_historical_data, scenario, seed)
            for seed in eval_seeds
        ]
        direct_summary = summarize_episode_rows(direct_rows, int(scenario.capacity))
        direct_vec_env.close()

        warm_run_name = f"{run_prefix}_{scenario.scenario_name}_ppo_beta_warm_start"
        warm_model, warm_vec_env, warm_run_dir, bc_metrics = train_ppo_beta_warm_start(
            baseline_policy_fn=inventory_policy,
            run_name=warm_run_name,
            historical_data=train_historical_data,
            capacity=int(scenario.capacity),
            train_seed=effective_train_seed,
            total_timesteps=int(total_timesteps),
            progress_bar=not bool(no_progress_bar),
            verbose=1,
            demo_episodes=int(demo_episodes),
            bc_epochs=int(bc_epochs),
            bc_batch_size=int(bc_batch_size),
            bc_learning_rate=float(bc_learning_rate),
            bc_entropy_coef=float(bc_entropy_coef),
        )
        warm_rows = [
            evaluate_learned_policy(warm_model, warm_vec_env, train_historical_data, scenario, seed)
            for seed in eval_seeds
        ]
        warm_summary = summarize_episode_rows(warm_rows, int(scenario.capacity))
        warm_vec_env.close()

    static_meta = {
        "best_static_prices": json.dumps(list(map(float, best_static_prices)), ensure_ascii=False),
        "baseline_selection_seeds": json.dumps(list(map(int, selection_seeds))),
        "baseline_top_k": int(candidate_count),
    }
    inventory_extra = {
        **inventory_meta,
        "baseline_selection_seeds": json.dumps(list(map(int, selection_seeds))),
        "baseline_top_k": int(candidate_count),
    }
    train_meta = {
        "train_seed": int(effective_train_seed),
        "eval_seed_count": int(len(eval_seeds)),
        "total_timesteps": int(total_timesteps),
    }
    warm_meta = {
        **train_meta,
        "demo_episodes": int(demo_episodes),
        "bc_epochs": int(bc_epochs),
        "bc_batch_size": int(bc_batch_size),
        "bc_learning_rate": float(bc_learning_rate),
        "bc_entropy_coef": float(bc_entropy_coef),
    }

    episode_rows = []
    episode_rows += add_episode_rows(
        static_rows,
        scenario=scenario,
        strategy_name="static_grid_best",
        algo="baseline",
        policy_type="baseline_static",
        eval_seeds=eval_seeds,
        extra=static_meta,
    )
    episode_rows += add_episode_rows(
        inventory_rows,
        scenario=scenario,
        strategy_name="inventory_protection_best",
        algo="baseline",
        policy_type="baseline_inventory",
        eval_seeds=eval_seeds,
        extra=inventory_extra,
    )
    episode_rows += add_episode_rows(
        direct_rows,
        scenario=scenario,
        strategy_name="ppo_beta_direct",
        algo="ppo_beta_direct",
        policy_type="learned",
        eval_seeds=eval_seeds,
        train_seed=effective_train_seed,
        total_timesteps=total_timesteps,
        run_name=direct_run_name,
        run_dir=str(direct_run_dir),
    )
    episode_rows += add_episode_rows(
        warm_rows,
        scenario=scenario,
        strategy_name="ppo_beta_warm_start",
        algo="ppo_beta_warm_start",
        policy_type="learned",
        eval_seeds=eval_seeds,
        train_seed=effective_train_seed,
        total_timesteps=total_timesteps,
        run_name=warm_run_name,
        run_dir=str(warm_run_dir),
        extra=warm_meta,
    )

    strategy_rows = [
        make_strategy_row(
            scenario=scenario,
            strategy_name="static_grid_best",
            algo="baseline",
            policy_type="baseline_static",
            summary=static_summary,
            extra=static_meta,
        ),
        make_strategy_row(
            scenario=scenario,
            strategy_name="inventory_protection_best",
            algo="baseline",
            policy_type="baseline_inventory",
            summary=inventory_summary,
            extra=inventory_extra,
        ),
        make_strategy_row(
            scenario=scenario,
            strategy_name="ppo_beta_direct",
            algo="ppo_beta_direct",
            policy_type="learned",
            summary=direct_summary,
            extra={**train_meta, "run_name": direct_run_name, "run_dir": str(direct_run_dir)},
        ),
        make_strategy_row(
            scenario=scenario,
            strategy_name="ppo_beta_warm_start",
            algo="ppo_beta_warm_start",
            policy_type="learned",
            summary=warm_summary,
            extra={**warm_meta, "run_name": warm_run_name, "run_dir": str(warm_run_dir)},
        ),
    ]

    summary_rows = [
        {
            **scenario_fields(scenario),
            "direct_revenue": float(direct_summary["episode_revenue_mean"]),
            "warm_start_revenue": float(warm_summary["episode_revenue_mean"]),
            "static_grid_best_revenue": float(static_summary["episode_revenue_mean"]),
            "inventory_protection_best_revenue": float(inventory_summary["episode_revenue_mean"]),
            "warm_start_vs_direct": float(
                warm_summary["episode_revenue_mean"] / max(1e-8, direct_summary["episode_revenue_mean"])
            ),
            "warm_start_vs_inventory": float(
                warm_summary["episode_revenue_mean"] / max(1e-8, inventory_summary["episode_revenue_mean"])
            ),
            "direct_vs_inventory": float(
                direct_summary["episode_revenue_mean"] / max(1e-8, inventory_summary["episode_revenue_mean"])
            ),
            "direct_acceptance_rate": float(direct_summary["episode_acceptance_rate_mean"]),
            "warm_start_acceptance_rate": float(warm_summary["episode_acceptance_rate_mean"]),
            "direct_full_day_rate": float(direct_summary["full_day_rate_mean"]),
            "warm_start_full_day_rate": float(warm_summary["full_day_rate_mean"]),
            "direct_avg_price_day0": float(direct_summary["avg_price_day0_mean"]),
            "direct_avg_price_day1": float(direct_summary["avg_price_day1_mean"]),
            "direct_avg_price_day2": float(direct_summary["avg_price_day2_mean"]),
            "warm_start_avg_price_day0": float(warm_summary["avg_price_day0_mean"]),
            "warm_start_avg_price_day1": float(warm_summary["avg_price_day1_mean"]),
            "warm_start_avg_price_day2": float(warm_summary["avg_price_day2_mean"]),
            "best_static_prices": static_meta["best_static_prices"],
            "inventory_base_prices": str(inventory_meta["inventory_base_prices"]),
            "inventory_scarcity_alpha": float(inventory_meta["inventory_scarcity_alpha"]),
            "baseline_selection_seeds": static_meta["baseline_selection_seeds"],
            "eval_seed_count": int(len(eval_seeds)),
            "train_seed": int(effective_train_seed),
            "total_timesteps": int(total_timesteps),
            "demo_episodes": int(demo_episodes),
            "bc_epochs": int(bc_epochs),
            "direct_run_dir": str(direct_run_dir),
            "warm_start_run_dir": str(warm_run_dir),
        }
    ]

    bc_rows = []
    for row in bc_metrics.to_dict(orient="records"):
        bc_rows.append(
            {
                **scenario_fields(scenario),
                "run_name": warm_run_name,
                "run_dir": str(warm_run_dir),
                **row,
            }
        )

    return episode_rows, strategy_rows, summary_rows, bc_rows


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenario_file, list(map(str, args.scenario_names)))
    train_historical_data = load_train_historical_data()
    selection_seeds = list(map(int, args.selection_seeds))
    eval_seeds = list(map(int, args.eval_seeds))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    experiment_root.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict[str, Any]] = []
    strategy_results: list[dict[str, Any]] = []
    summary_results: list[dict[str, Any]] = []
    bc_results: list[dict[str, Any]] = []

    for scenario in scenarios:
        episode_rows, strategy_rows, summary_rows, bc_rows = run_scenario(
            scenario=scenario,
            train_historical_data=train_historical_data,
            selection_seeds=selection_seeds,
            eval_seeds=eval_seeds,
            price_grid=list(map(float, args.price_grid)),
            baseline_top_k=int(args.baseline_top_k),
            train_seed=args.train_seed,
            total_timesteps=int(args.total_timesteps),
            run_prefix=str(args.run_prefix),
            no_progress_bar=bool(args.no_progress_bar),
            demo_episodes=int(args.demo_episodes),
            bc_epochs=int(args.bc_epochs),
            bc_batch_size=int(args.bc_batch_size),
            bc_learning_rate=float(args.bc_learning_rate),
            bc_entropy_coef=float(args.bc_entropy_coef),
        )
        episode_results.extend(episode_rows)
        strategy_results.extend(strategy_rows)
        summary_results.extend(summary_rows)
        bc_results.extend(bc_rows)
        summary = summary_rows[0]
        print(
            f"[{scenario.scenario_name}] direct={float(summary['direct_revenue']):.2f}, "
            f"warm={float(summary['warm_start_revenue']):.2f}, "
            f"inventory={float(summary['inventory_protection_best_revenue']):.2f}, "
            f"warm/direct={float(summary['warm_start_vs_direct']):.4f}"
        )

    episode_df = pd.DataFrame(episode_results).sort_values(
        ["scenario_name", "strategy_name", "eval_seed"]
    )
    strategy_df = pd.DataFrame(strategy_results).sort_values(["scenario_name", "strategy_name"])
    summary_df = pd.DataFrame(summary_results).sort_values(["scenario_name"])
    bc_df = pd.DataFrame(bc_results).sort_values(["scenario_name", "epoch"])

    episode_csv = experiment_root / "eval_episode_results.csv"
    strategy_csv = experiment_root / "strategy_results.csv"
    summary_csv = experiment_root / "warm_start_summary.csv"
    bc_csv = experiment_root / "bc_pretrain_metrics.csv"
    episode_df.to_csv(episode_csv, index=False)
    strategy_df.to_csv(strategy_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    bc_df.to_csv(bc_csv, index=False)

    split_metadata = get_data_split_metadata(train_historical_data, train_historical_data)
    metadata = {
        "scenarios": [scenario_fields(scenario) for scenario in scenarios],
        "distribution_mode": "same_distribution_train_years",
        "selection_seeds": selection_seeds,
        "eval_seeds": eval_seeds,
        "total_timesteps": int(args.total_timesteps),
        "demo_episodes": int(args.demo_episodes),
        "bc_epochs": int(args.bc_epochs),
        "bc_batch_size": int(args.bc_batch_size),
        "bc_learning_rate": float(args.bc_learning_rate),
        "bc_entropy_coef": float(args.bc_entropy_coef),
        "price_grid": list(map(float, args.price_grid)),
        "baseline_top_k": int(args.baseline_top_k),
        **split_metadata,
        "episode_results_csv": str(episode_csv),
        "strategy_results_csv": str(strategy_csv),
        "warm_start_summary_csv": str(summary_csv),
        "bc_pretrain_metrics_csv": str(bc_csv),
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"实验完成，逐 seed 测试结果: {episode_csv}")
    print(f"策略汇总表: {strategy_csv}")
    print(f"warm-start 对比表: {summary_csv}")
    print(f"BC 预训练指标: {bc_csv}")


if __name__ == "__main__":
    main()
