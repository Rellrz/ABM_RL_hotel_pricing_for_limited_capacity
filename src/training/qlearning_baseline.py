"""Q-learning runner（单Q表，按stage遍历每日8桶动作）。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.agent.qlearning_agent import QLearningAgent
from src.utils.common import build_cem_flat_state
from configs.experiment2 import Experiment2Config
from src.environment.bucket_pricing_simulator import BucketPricingSimulator
from src.evaluation.policy_evaluator import evaluate_policy


def run_qlearning(config: Experiment2Config, historical_data) -> tuple[List[Dict], List[Dict]]:
    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []
    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="Q-learning Seeds", unit="seed"):
            tqdm.write(f"[Q-learning] Seed {seed} start")
            seed_train_records, seed_eval_records, _seed = _run_single_seed(config, historical_data, seed, show_progress=True)
            all_train_records.extend(seed_train_records)
            all_eval_records.extend(seed_eval_records)
            tqdm.write(f"[Q-learning] Seed {_seed} done: ep={len(seed_train_records)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed, config, historical_data, seed, True))

        with tqdm(total=len(futures), desc="Q-learning Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                seed_train_records, seed_eval_records, seed = fut.result()
                all_train_records.extend(seed_train_records)
                all_eval_records.extend(seed_eval_records)
                pbar.update(1)
                tqdm.write(f"[Q-learning] Seed {seed} done: ep={len(seed_train_records)}")
    return all_train_records, all_eval_records


def _run_single_seed(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
) -> tuple[List[Dict], List[Dict], int]:
    train_records: List[Dict] = []
    eval_records: List[Dict] = []
    agent = QLearningAgent(config=config, seed=seed)
    sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
    sim.reset()
    init_actions = []
    for sid in range(sim.n_stages):
        st0 = sim.get_state_by_stage(sid)
        s0 = build_cem_flat_state(st0, sid)
        a0_idx = agent.select_action(s0, deterministic=False)
        a0 = config.q_action_grid[a0_idx]
        init_actions.append((float(a0[0]), float(a0[1])))
    sim.initialize_episode_decisions(init_actions)
    steps = 0
    done = False
    episode_idx = 0
    episode_hotel_revenue = 0.0
    episode_ota_profit = 0.0
    pbar = tqdm(
        total=config.train_steps,
        desc=f"Q Seed {seed}",
        unit="step",
        leave=False,
        disable=not show_progress,
    )
    while steps < config.train_steps:
        if done:
            sim.reset()
            init_actions = []
            for sid in range(sim.n_stages):
                st0 = sim.get_state_by_stage(sid)
                s0 = build_cem_flat_state(st0, sid)
                a0_idx = agent.select_action(s0, deterministic=False)
                a0 = config.q_action_grid[a0_idx]
                init_actions.append((float(a0[0]), float(a0[1])))
            sim.initialize_episode_decisions(init_actions)
            done = False

        stage_actions = []
        for sid in range(sim.n_stages):
            st = sim.get_state_by_stage(sid)
            s_idx = build_cem_flat_state(st, sid)
            a_idx = agent.select_action(s_idx, deterministic=False)
            a = config.q_action_grid[a_idx]
            stage_actions.append((float(a[0]), float(a[1])))

        out = sim.step_day(stage_actions)
        episode_hotel_revenue += float(out.reward_hotel)
        episode_ota_profit += float(out.reward_ota)
        update_events = out.info.get("update_events", [])
        for ev in update_events:
            s = build_cem_flat_state(ev.state, int(ev.state.get("stage_id", 0)))
            s_next = build_cem_flat_state(ev.next_state, int(ev.next_state.get("stage_id", 0)))
            a_pair = np.asarray(ev.action_pair, dtype=np.float64)
            # 将连续动作映射回离散动作网格索引
            dist = np.sum((config.q_action_grid - a_pair.reshape(1, 2)) ** 2, axis=1)
            a = int(np.argmin(dist))
            r = float(ev.reward)
            agent.update(s, a, r, s_next, bool(ev.done))

        steps += 1
        pbar.update(1)
        done = out.done
        if done:
            episode_idx += 1
            train_records.append(
                {
                    "Algorithm": "Q-learning",
                    "Seed": seed,
                    "Episode": episode_idx,
                    "EpisodeHotelRevenue": float(episode_hotel_revenue),
                    "EpisodeOTAProfit": float(episode_ota_profit),
                    "EpisodeSystemProfit": float(episode_hotel_revenue + episode_ota_profit),
                    "EpisodeRevenue": float(episode_hotel_revenue),
                }
            )
            episode_hotel_revenue = 0.0
            episode_ota_profit = 0.0

        pbar.set_postfix({"ep": episode_idx, "day": sim.day})

    pbar.close()

    def stage_policy_fn(stage_id: int, st: dict):
        s_idx = build_cem_flat_state(st, stage_id)
        a_idx = agent.select_action(s_idx, deterministic=True)
        a = config.q_action_grid[a_idx]
        return float(a[0]), float(a[1])

    eval_rewards = evaluate_policy(
        config=config,
        historical_data=historical_data,
        seed=seed + 300_000,
        stage_policy_fn=stage_policy_fn,
        n_episodes=config.post_eval_episodes,
    )
    for idx, rew in enumerate(eval_rewards, start=1):
        eval_records.append(
            {
                "Algorithm": "Q-learning",
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalHotelRevenue": float(rew["EvalHotelRevenue"]),
                "EvalOTAProfit": float(rew["EvalOTAProfit"]),
                "EvalSystemProfit": float(rew["EvalSystemProfit"]),
                "EvalRevenue": float(rew["EvalHotelRevenue"]),
            }
        )
    return train_records, eval_records, seed
