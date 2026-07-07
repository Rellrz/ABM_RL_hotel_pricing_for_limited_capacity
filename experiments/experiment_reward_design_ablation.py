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

from configs.config import ABM_CONFIG, ENV_CONFIG, PATH_CONFIG, PPO_CONFIG
from src.baseline.pricing_baselines import (
    add_derived_metrics,
    get_inventory_protection_policy,
    get_static_policy,
    rank_static_candidates,
    search_inventory_protection_best,
    summarize_episode_rows,
)
from src.environment.abm_customer_model import load_filtered_historical_data
from src.environment.gym_hotel_env import GymHotelPricingEnv
from src.training.algorithm_registry import get_algorithm_runner
from src.training.train_ppo import EpisodeMetricsAggregator


DEFAULT_SCENARIO = {
    "scenario_name": "scenario_b_cap20_flex075_lam48",
    "capacity": 20,
    "flexible_customer_share": 0.75,
    "lambda_day_mismatch_flex": 48.0,
}
DEFAULT_EVAL_SEEDS = [142, 143, 144]
DEFAULT_PRICE_GRID = [100.0, 150.0, 200.0]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_name: str
    capacity: int
    flexible_customer_share: float
    lambda_day_mismatch_flex: float


@dataclass(frozen=True)
class RewardSpec:
    name: str
    env_overrides: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="current/no-penalty/weighted-scarcity reward 设计验证实验")
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--weighted-scarcity-coef", type=float, default=9000.0)
    parser.add_argument("--price-grid", nargs="+", type=float, default=DEFAULT_PRICE_GRID)
    parser.add_argument("--baseline-top-k", type=int, default=10)
    parser.add_argument("--run-prefix", type=str, default="reward_design_ablation")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


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


def scenario_from_default() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_name=str(DEFAULT_SCENARIO["scenario_name"]),
        capacity=int(DEFAULT_SCENARIO["capacity"]),
        flexible_customer_share=float(DEFAULT_SCENARIO["flexible_customer_share"]),
        lambda_day_mismatch_flex=float(DEFAULT_SCENARIO["lambda_day_mismatch_flex"]),
    )


def reward_specs(weighted_scarcity_coef: float) -> list[RewardSpec]:
    return [
        RewardSpec(
            name="current_standard",
            env_overrides={
                "reward_mode": "standard",
                "full_capacity_penalty": float(ENV_CONFIG.full_capacity_penalty),
                "scarcity_threshold_ratio": float(ENV_CONFIG.scarcity_threshold_ratio),
                "scarcity_penalty_coef": float(ENV_CONFIG.scarcity_penalty_coef),
                "scarcity_penalty_weights": [1.0, 1.0, 1.0],
            },
        ),
        RewardSpec(
            name="no_penalty",
            env_overrides={
                "reward_mode": "no_penalty",
            },
        ),
        RewardSpec(
            name=f"weighted_scarcity_{int(weighted_scarcity_coef)}",
            env_overrides={
                "reward_mode": "weighted_scarcity",
                "full_capacity_penalty": 0.0,
                "scarcity_threshold_ratio": float(ENV_CONFIG.scarcity_threshold_ratio),
                "scarcity_penalty_coef": float(weighted_scarcity_coef),
                "scarcity_penalty_weights": [0.0, 0.5, 1.0],
            },
        ),
    ]


def scenario_fields(scenario: ScenarioSpec) -> dict[str, Any]:
    return {
        "scenario_name": scenario.scenario_name,
        "capacity": int(scenario.capacity),
        "flexible_customer_share": float(scenario.flexible_customer_share),
        "lambda_day_mismatch_flex": float(scenario.lambda_day_mismatch_flex),
    }


def evaluate_manual_policy_with_reward(
    policy_fn,
    historical_data: pd.DataFrame,
    scenario: ScenarioSpec,
    eval_seed: int,
    env_overrides: dict[str, Any],
) -> dict[str, float]:
    env = GymHotelPricingEnv(
        historical_data=historical_data,
        seed=int(eval_seed),
        capacity=int(scenario.capacity),
        **env_overrides,
    )
    aggregator = EpisodeMetricsAggregator()
    episode_reward = 0.0
    obs, _ = env.reset(seed=int(eval_seed))
    done = False
    while not done:
        prices = policy_fn(obs)
        midpoint = 0.5 * (float(ENV_CONFIG.price_min) + float(ENV_CONFIG.price_max))
        half_range = 0.5 * (float(ENV_CONFIG.price_max) - float(ENV_CONFIG.price_min))
        action = (prices - midpoint) / max(1e-8, half_range)
        obs, reward, terminated, truncated, info = env.step(action)
        aggregator.update(info)
        episode_reward += float(reward)
        done = bool(terminated or truncated)
    env.close()
    metrics = aggregator.summary()
    metrics["episode_reward"] = float(episode_reward)
    return metrics


def evaluate_learned_policy(
    *,
    model,
    train_vec_env,
    historical_data: pd.DataFrame,
    scenario: ScenarioSpec,
    eval_seed: int,
    env_overrides: dict[str, Any],
) -> dict[str, float]:
    build_eval_env_fn = get_algorithm_runner("ppo_beta")["build_eval_env"]
    eval_env = build_eval_env_fn(
        train_vec_env=train_vec_env,
        historical_data=historical_data,
        seed=int(eval_seed),
        capacity=int(scenario.capacity),
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


def add_episode_rows(
    rows: list[dict[str, float]],
    *,
    scenario: ScenarioSpec,
    reward_spec: RewardSpec,
    strategy_name: str,
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
            "reward_name": reward_spec.name,
            "reward_mode": str(reward_spec.env_overrides.get("reward_mode", "")),
            "reward_env_overrides": json.dumps(reward_spec.env_overrides, ensure_ascii=False),
            "strategy_name": strategy_name,
            "algo": "ppo_beta" if policy_type == "learned" else "baseline",
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
    reward_spec: RewardSpec,
    strategy_name: str,
    policy_type: str,
    summary: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **scenario_fields(scenario),
        "reward_name": reward_spec.name,
        "reward_mode": str(reward_spec.env_overrides.get("reward_mode", "")),
        "reward_env_overrides": json.dumps(reward_spec.env_overrides, ensure_ascii=False),
        "strategy_name": strategy_name,
        "algo": "ppo_beta" if policy_type == "learned" else "baseline",
        "policy_type": policy_type,
    }
    row.update(summary)
    if extra:
        row.update(extra)
    return row


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario = scenario_from_default()
    historical_data = load_filtered_historical_data()
    eval_seeds = list(map(int, args.eval_seeds))
    price_grid = list(map(float, args.price_grid))
    effective_train_seed = int(PPO_CONFIG.seed if args.train_seed is None else args.train_seed)
    effective_total_timesteps = int(args.total_timesteps)

    episode_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    reward_summary_rows: list[dict[str, Any]] = []

    with apply_abm_scenario(scenario):
        ranked_static = rank_static_candidates(
            historical_data=historical_data,
            capacity=int(scenario.capacity),
            eval_seeds=eval_seeds,
            price_grid=price_grid,
        )
        _, best_static_prices, _ = ranked_static[0]
        static_policy = get_static_policy(best_static_prices)
        candidate_count = max(1, min(int(args.baseline_top_k), len(ranked_static)))
        candidate_prices = [prices for _, prices, _ in ranked_static[:candidate_count]]
        _, inventory_meta = search_inventory_protection_best(
            historical_data=historical_data,
            capacity=int(scenario.capacity),
            eval_seeds=eval_seeds,
            candidate_prices=candidate_prices,
        )
        inventory_policy = get_inventory_protection_policy(
            base_prices=tuple(json.loads(str(inventory_meta["inventory_base_prices"]))),
            scarcity_alpha=float(inventory_meta["inventory_scarcity_alpha"]),
            capacity=int(scenario.capacity),
        )

        for reward_spec in reward_specs(float(args.weighted_scarcity_coef)):
            static_rows = [
                evaluate_manual_policy_with_reward(
                    static_policy,
                    historical_data,
                    scenario,
                    seed,
                    reward_spec.env_overrides,
                )
                for seed in eval_seeds
            ]
            inventory_rows = [
                evaluate_manual_policy_with_reward(
                    inventory_policy,
                    historical_data,
                    scenario,
                    seed,
                    reward_spec.env_overrides,
                )
                for seed in eval_seeds
            ]
            static_summary = summarize_episode_rows(static_rows, int(scenario.capacity))
            inventory_summary = summarize_episode_rows(inventory_rows, int(scenario.capacity))
            static_extra = {
                "best_static_prices": json.dumps(list(map(float, best_static_prices)), ensure_ascii=False),
                "baseline_top_k": int(candidate_count),
            }
            inventory_extra = {**inventory_meta, "baseline_top_k": int(candidate_count)}

            strategy_rows.append(
                make_strategy_row(
                    scenario=scenario,
                    reward_spec=reward_spec,
                    strategy_name="static_grid_best",
                    policy_type="baseline_static",
                    summary=static_summary,
                    extra=static_extra,
                )
            )
            strategy_rows.append(
                make_strategy_row(
                    scenario=scenario,
                    reward_spec=reward_spec,
                    strategy_name="inventory_protection_best",
                    policy_type="baseline_inventory",
                    summary=inventory_summary,
                    extra=inventory_extra,
                )
            )
            episode_rows.extend(
                add_episode_rows(
                    static_rows,
                    scenario=scenario,
                    reward_spec=reward_spec,
                    strategy_name="static_grid_best",
                    policy_type="baseline_static",
                    eval_seeds=eval_seeds,
                    extra=static_extra,
                )
            )
            episode_rows.extend(
                add_episode_rows(
                    inventory_rows,
                    scenario=scenario,
                    reward_spec=reward_spec,
                    strategy_name="inventory_protection_best",
                    policy_type="baseline_inventory",
                    eval_seeds=eval_seeds,
                    extra=inventory_extra,
                )
            )

            run_name = f"{args.run_prefix}_{scenario.scenario_name}_ppo_beta_{reward_spec.name}"
            train_single_run_fn = get_algorithm_runner("ppo_beta")["train_single_run"]
            model, train_vec_env, run_dir = train_single_run_fn(
                run_name=run_name,
                historical_data=historical_data,
                capacity=int(scenario.capacity),
                train_seed=effective_train_seed,
                total_timesteps=effective_total_timesteps,
                progress_bar=not bool(args.no_progress_bar),
                verbose=1,
                env_overrides=reward_spec.env_overrides,
            )
            learned_rows = [
                evaluate_learned_policy(
                    model=model,
                    train_vec_env=train_vec_env,
                    historical_data=historical_data,
                    scenario=scenario,
                    eval_seed=seed,
                    env_overrides=reward_spec.env_overrides,
                )
                for seed in eval_seeds
            ]
            learned_summary = summarize_episode_rows(learned_rows, int(scenario.capacity))
            train_vec_env.close()

            strategy_rows.append(
                make_strategy_row(
                    scenario=scenario,
                    reward_spec=reward_spec,
                    strategy_name="ppo_beta",
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
            )
            episode_rows.extend(
                add_episode_rows(
                    learned_rows,
                    scenario=scenario,
                    reward_spec=reward_spec,
                    strategy_name="ppo_beta",
                    policy_type="learned",
                    eval_seeds=eval_seeds,
                    train_seed=effective_train_seed,
                    total_timesteps=effective_total_timesteps,
                    run_name=run_name,
                    run_dir=str(run_dir),
                )
            )
            reward_summary_rows.append(
                {
                    **scenario_fields(scenario),
                    "reward_name": reward_spec.name,
                    "reward_mode": str(reward_spec.env_overrides["reward_mode"]),
                    "learned_revenue": float(learned_summary["episode_revenue_mean"]),
                    "learned_reward": float(learned_summary["episode_reward_mean"]),
                    "learned_penalty": float(learned_summary["episode_penalty_mean"]),
                    "learned_full_day_rate": float(learned_summary["full_day_rate_mean"]),
                    "learned_full_rate_day0": float(learned_summary["full_rate_day0_mean"]),
                    "learned_full_rate_day1": float(learned_summary["full_rate_day1_mean"]),
                    "learned_full_rate_day2": float(learned_summary["full_rate_day2_mean"]),
                    "learned_avg_rejected_day0": float(learned_summary["avg_rejected_day0_mean"]),
                    "learned_avg_rejected_day1": float(learned_summary["avg_rejected_day1_mean"]),
                    "learned_avg_rejected_day2": float(learned_summary["avg_rejected_day2_mean"]),
                    "learned_avg_price_day0": float(learned_summary["avg_price_day0_mean"]),
                    "learned_avg_price_day1": float(learned_summary["avg_price_day1_mean"]),
                    "learned_avg_price_day2": float(learned_summary["avg_price_day2_mean"]),
                    "inventory_revenue": float(inventory_summary["episode_revenue_mean"]),
                    "static_revenue": float(static_summary["episode_revenue_mean"]),
                    "learned_vs_inventory_revenue": float(
                        learned_summary["episode_revenue_mean"] / max(1e-8, inventory_summary["episode_revenue_mean"])
                    ),
                    "learned_vs_static_revenue": float(
                        learned_summary["episode_revenue_mean"] / max(1e-8, static_summary["episode_revenue_mean"])
                    ),
                    "train_seed": effective_train_seed,
                    "eval_seed_count": int(len(eval_seeds)),
                    "total_timesteps": effective_total_timesteps,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                }
            )
            print(
                f"[{reward_spec.name}] revenue={learned_summary['episode_revenue_mean']:.2f}, "
                f"full_day_rate={learned_summary['full_day_rate_mean']:.4f}, "
                f"avg_price=({learned_summary['avg_price_day0_mean']:.1f}, "
                f"{learned_summary['avg_price_day1_mean']:.1f}, "
                f"{learned_summary['avg_price_day2_mean']:.1f})"
            )

    return pd.DataFrame(episode_rows), pd.DataFrame(strategy_rows), pd.DataFrame(reward_summary_rows)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = PATH_CONFIG.output_root / "experiments" / f"{args.run_prefix}_{timestamp}"
    experiment_root.mkdir(parents=True, exist_ok=True)

    episode_df, strategy_df, summary_df = run_experiment(args)
    episode_df = episode_df.sort_values(["reward_name", "strategy_name", "eval_seed"]).reset_index(drop=True)
    strategy_df = strategy_df.sort_values(["reward_name", "strategy_name"]).reset_index(drop=True)
    summary_df = summary_df.sort_values(["reward_name"]).reset_index(drop=True)

    episode_csv = experiment_root / "eval_episode_results.csv"
    strategy_csv = experiment_root / "strategy_results.csv"
    summary_csv = experiment_root / "reward_summary.csv"
    episode_df.to_csv(episode_csv, index=False)
    strategy_df.to_csv(strategy_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "scenario": DEFAULT_SCENARIO,
        "train_seed": args.train_seed,
        "eval_seeds": list(map(int, args.eval_seeds)),
        "total_timesteps": int(args.total_timesteps),
        "weighted_scarcity_coef": float(args.weighted_scarcity_coef),
        "episode_results_csv": str(episode_csv),
        "strategy_results_csv": str(strategy_csv),
        "reward_summary_csv": str(summary_csv),
    }
    with open(experiment_root / "experiment_summary.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"实验完成，逐 seed 测试结果: {episode_csv}")
    print(f"策略汇总表: {strategy_csv}")
    print(f"reward 对比表: {summary_csv}")


if __name__ == "__main__":
    main()
