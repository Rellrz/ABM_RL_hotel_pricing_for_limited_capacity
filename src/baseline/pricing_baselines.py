from __future__ import annotations

from itertools import product
import json
from typing import Callable

import numpy as np
import pandas as pd

from configs.config import ENV_CONFIG
from src.environment.gym_hotel_env import GymHotelPricingEnv
from src.training.train_ppo import EpisodeMetricsAggregator


DEFAULT_PRICE_GRID = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
DEFAULT_INVENTORY_ALPHA = [0.0, 50.0, 100.0, 150.0, 200.0]

METRIC_BASE_NAMES = [
    "episode_revenue",
    "episode_reward",
    "episode_penalty",
    "episode_full_penalty",
    "episode_scarcity_penalty",
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
    "avg_rejected_day0",
    "avg_rejected_day1",
    "avg_rejected_day2",
    "revenue_per_arrival",
    "revenue_per_capacity_day",
]


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


__all__ = [
    "DEFAULT_INVENTORY_ALPHA",
    "DEFAULT_PRICE_GRID",
    "evaluate_manual_policy",
    "evaluate_policy_over_seeds",
    "get_inventory_protection_policy",
    "get_static_policy",
    "get_weekday_weekend_static_policy",
    "rank_static_candidates",
    "search_inventory_protection_best",
    "search_weekday_weekend_static_best",
]
