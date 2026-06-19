"""统一评估器：训练结束后批量评估。"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from configs.experiment2 import Experiment2Config
from src.environment.bucket_pricing_simulator import BucketPricingSimulator
from src.environment.ppo_bucket_env import PPOBucketEnv


StagePolicyFn = Callable[[int, dict], Tuple[float, float]]


def evaluate_policy(
    config: Experiment2Config,
    historical_data,
    seed: int,
    stage_policy_fn: StagePolicyFn,
    n_episodes: int,
) -> List[Dict[str, float]]:
    rewards: List[Dict[str, float]] = []
    for ep in range(n_episodes):
        sim = BucketPricingSimulator(config=config, seed=seed * 1000 + ep, historical_data=historical_data)
        sim.reset()
        total_hotel = 0.0
        total_ota = 0.0
        done = False
        while not done:
            stage_actions = []
            for sid in range(sim.n_stages):
                st = sim.get_state_by_stage(sid)
                stage_actions.append(stage_policy_fn(sid, st))
            out = sim.step_day(stage_actions)
            total_hotel += out.reward_hotel
            total_ota += out.reward_ota
            done = out.done
        rewards.append(
            {
                "EvalHotelRevenue": float(total_hotel),
                "EvalOTAProfit": float(total_ota),
                "EvalSystemProfit": float(total_hotel + total_ota),
            }
        )
    return rewards


def evaluate_ppo_model(
    config: Experiment2Config,
    historical_data,
    seed: int,
    model,
    n_episodes: int,
    vec_normalizer=None,
) -> List[Dict[str, float]]:
    import copy

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    rewards: List[Dict[str, float]] = []
    for ep in range(n_episodes):
        env_seed = seed * 1000 + ep

        def _make_env():
            return PPOBucketEnv(config=config, seed=env_seed, historical_data=historical_data)

        base_env = DummyVecEnv([_make_env])
        if vec_normalizer is not None:
            eval_env = VecNormalize(
                base_env,
                training=False,
                norm_obs=bool(getattr(vec_normalizer, "norm_obs", True)),
                norm_reward=False,
                clip_obs=float(getattr(vec_normalizer, "clip_obs", 10.0)),
                clip_reward=float(getattr(vec_normalizer, "clip_reward", 10.0)),
                gamma=float(getattr(vec_normalizer, "gamma", config.ppo_gamma)),
            )
            if getattr(vec_normalizer, "obs_rms", None) is not None:
                eval_env.obs_rms = copy.deepcopy(vec_normalizer.obs_rms)
        else:
            eval_env = base_env

        obs = eval_env.reset()
        done = False
        total_hotel = 0.0
        total_ota = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards_arr, dones, infos = eval_env.step(action)
            del rewards_arr
            info = infos[0]
            reward_hotel_by_stage = info.get("reward_hotel_by_stage", [])
            total_hotel += float(sum(reward_hotel_by_stage))
            reward_ota_by_stage = info.get("reward_ota_by_stage", [])
            total_ota += float(sum(reward_ota_by_stage))
            done = bool(dones[0])
        rewards.append(
            {
                "EvalHotelRevenue": float(total_hotel),
                "EvalOTAProfit": float(total_ota),
                "EvalSystemProfit": float(total_hotel + total_ota),
            }
        )
        eval_env.close()
    return rewards
