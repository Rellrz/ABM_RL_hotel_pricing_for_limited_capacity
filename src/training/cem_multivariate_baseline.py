"""Multivariate CEM 基线：全协方差矩阵的交叉熵方法。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from configs.experiment2 import Experiment2Config
from src.environment.bucket_pricing_simulator import BucketPricingSimulator
from src.evaluation.policy_evaluator import evaluate_policy
from src.algorithms.multivariate_cem import MultivariateCrossEntropyMethod
from src.utils.common import build_cem_state_key


def _run_single_seed_multivariate(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    cem = MultivariateCrossEntropyMethod(
        n_states=config.q_n_states,
        action_mins=(config.online_price_min, config.offline_price_min),
        action_maxs=(config.online_price_max, config.offline_price_max),
        discount_factor=0.99,
        n_samples=config.cem_n_samples,
        elite_frac=config.cem_elite_frac,
        initial_std=config.cem_initial_std,
        min_std=config.cem_min_std,
        std_decay=config.cem_std_decay,
        memory_size=config.cem_memory_size,
    )
    sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
    sim.reset()
    init_actions = []
    for sid in range(sim.n_stages):
        st0 = sim.get_state_by_stage(sid)
        s0 = build_cem_state_key(st0, sid)
        a0 = cem.select_action(s0, deterministic=False)
        init_actions.append((float(a0[0]), float(a0[1])))
    sim.initialize_episode_decisions(init_actions)
    train_records: List[Dict] = []
    eval_records: List[Dict] = []
    steps = 0
    done = False
    episode_idx = 0
    episode_hotel_revenue = 0.0
    episode_ota_profit = 0.0
    pbar = tqdm(
        total=config.train_steps,
        desc=f"MV-CEM Seed {seed}",
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
                s0 = build_cem_state_key(st0, sid)
                a0 = cem.select_action(s0, deterministic=False)
                init_actions.append((float(a0[0]), float(a0[1])))
            sim.initialize_episode_decisions(init_actions)
            done = False

        actions = []
        for sid in range(sim.n_stages):
            st = sim.get_state_by_stage(sid)
            s_idx = build_cem_state_key(st, sid)
            act = cem.select_action(s_idx, deterministic=False)
            actions.append((float(act[0]), float(act[1])))

        out = sim.step_day(actions)
        episode_hotel_revenue += float(out.reward_hotel)
        episode_ota_profit += float(out.reward_ota)
        update_events = out.info.get("update_events", [])
        for ev in update_events:
            s = build_cem_state_key(ev.state, int(ev.state.get("stage_id", 0)))
            s_next = build_cem_state_key(ev.next_state, int(ev.next_state.get("stage_id", 0)))
            a = np.asarray(ev.action_pair, dtype=np.float64)
            cem.update(s, a, float(ev.reward), s_next, bool(ev.done))

        steps += 1
        pbar.update(1)
        done = out.done
        if done:
            episode_idx += 1
            train_records.append(
                {
                    "Algorithm": "Multivariate CEM",
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

        if config.update_frequency > 0 and (sim.day % config.update_frequency == 0):
            cem.end_episode()
        if done and (config.update_frequency <= 0 or (sim.day % config.update_frequency != 0)):
            cem.end_episode()
        pbar.set_postfix({"ep": episode_idx, "day": sim.day})
    pbar.close()

    def stage_policy_fn(stage_id: int, st: dict):
        s_idx = build_cem_state_key(st, stage_id)
        action = cem.select_action(s_idx, deterministic=True)
        return float(action[0]), float(action[1])

    eval_rewards = evaluate_policy(
        config=config,
        historical_data=historical_data,
        seed=seed + 100_000,
        stage_policy_fn=stage_policy_fn,
        n_episodes=config.post_eval_episodes,
    )
    for idx, rew in enumerate(eval_rewards, start=1):
        eval_records.append(
            {
                "Algorithm": "Multivariate CEM",
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalHotelRevenue": float(rew["EvalHotelRevenue"]),
                "EvalOTAProfit": float(rew["EvalOTAProfit"]),
                "EvalSystemProfit": float(rew["EvalSystemProfit"]),
                "EvalRevenue": float(rew["EvalHotelRevenue"]),
            }
        )
    return train_records, eval_records


def run_multivariate_cem(
    config: Experiment2Config,
    historical_data,
) -> Tuple[List[Dict], List[Dict]]:
    """Multivariate CEM 基线主入口，支持多 seed 并行。"""
    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []

    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="MV-CEM Seeds", unit="seed"):
            train_rec, eval_rec = _run_single_seed_multivariate(
                config, historical_data, seed, show_progress=True,
            )
            all_train_records.extend(train_rec)
            all_eval_records.extend(eval_rec)
            tqdm.write(f"[MV-CEM] Seed {seed} done: train_ep={len(train_rec)} eval_ep={len(eval_rec)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(
                ex.submit(_run_single_seed_multivariate, config, historical_data, seed, True)
            )

        with tqdm(total=len(futures), desc="MV-CEM Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                train_rec, eval_rec = fut.result()
                all_train_records.extend(train_rec)
                all_eval_records.extend(eval_rec)
                pbar.update(1)

    return all_train_records, all_eval_records
