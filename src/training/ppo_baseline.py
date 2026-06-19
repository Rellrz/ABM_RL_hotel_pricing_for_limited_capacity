"""PPO runner（SB3）。"""

from __future__ import annotations

import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from configs.experiment2 import Experiment2Config
from src.evaluation.policy_evaluator import evaluate_ppo_model
from src.environment.ppo_bucket_env import PPOBucketEnv


def run_ppo(config: Experiment2Config, historical_data) -> tuple[List[Dict], List[Dict]]:
    if importlib.util.find_spec("stable_baselines3") is None:  # pragma: no cover
        raise RuntimeError(
            "未检测到 stable-baselines3，请先在abm环境安装：pip install stable-baselines3"
        )

    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []
    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="PPO Seeds", unit="seed"):
            tqdm.write(f"[PPO] Seed {seed} start")
            train_records, eval_records, _seed = _run_single_seed(config, historical_data, seed, show_progress=True)
            all_train_records.extend(train_records)
            all_eval_records.extend(eval_records)
            tqdm.write(f"[PPO] Seed {_seed} done: ep={len(train_records)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed, config, historical_data, seed, True))

        with tqdm(total=len(futures), desc="PPO Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                train_records, eval_records, seed = fut.result()
                all_train_records.extend(train_records)
                all_eval_records.extend(eval_records)
                pbar.update(1)
                tqdm.write(f"[PPO] Seed {seed} done: ep={len(train_records)}")
    return all_train_records, all_eval_records


def run_ppo_single_seed(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
    algorithm_name: str = "PPO",
) -> tuple[List[Dict], List[Dict], int]:
    return _run_single_seed(
        config=config,
        historical_data=historical_data,
        seed=seed,
        show_progress=show_progress,
        algorithm_name=algorithm_name,
    )


def _run_single_seed(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
    algorithm_name: str = "PPO",
) -> tuple[List[Dict], List[Dict], int]:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class EpisodeRevenueCallback(BaseCallback):
        def __init__(self, target_episodes: int):
            super().__init__()
            self.target_episodes = int(target_episodes)
            self.episode_hotel_rewards: List[float] = []
            self.episode_ota_profits: List[float] = []
            self._current_hotel_reward = 0.0
            self._current_ota_profit = 0.0

        def _on_step(self) -> bool:
            dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
            infos = self.locals.get("infos", [])
            for i, done in enumerate(dones):
                info = infos[i] if i < len(infos) else {}
                reward_hotel_by_stage = info.get("reward_hotel_by_stage", [])
                self._current_hotel_reward += float(sum(reward_hotel_by_stage))
                reward_ota_by_stage = info.get("reward_ota_by_stage", [])
                self._current_ota_profit += float(sum(reward_ota_by_stage))
                if bool(done):
                    self.episode_hotel_rewards.append(float(self._current_hotel_reward))
                    self.episode_ota_profits.append(float(self._current_ota_profit))
                    self._current_hotel_reward = 0.0
                    self._current_ota_profit = 0.0
            if len(self.episode_hotel_rewards) >= self.target_episodes:
                return False
            return True

    train_records: List[Dict] = []
    eval_records: List[Dict] = []
    def _make_env():
        return PPOBucketEnv(config=config, seed=seed, historical_data=historical_data)

    vec_env = VecNormalize(
        DummyVecEnv([_make_env]),
        training=True,
        norm_obs=bool(config.ppo_norm_obs),
        norm_reward=bool(config.ppo_norm_reward),
        clip_obs=float(config.ppo_clip_obs),
        clip_reward=float(config.ppo_clip_reward),
        gamma=float(config.ppo_gamma),
    )
    requested_device = str(getattr(config, "ppo_device", "auto")).strip().lower()
    if requested_device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            requested_device = "cpu"
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            requested_device = "cpu"
    elif requested_device not in {"auto", "cpu"}:
        requested_device = "auto"

    print(f"[PPO] Device: {requested_device}")
    
    policy_kwargs = dict(
        net_arch=list(config.ppo_net_arch),
        log_std_init=float(getattr(config, "ppo_log_std_init", -0.2)),
    )
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.ppo_learning_rate,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        ent_coef=config.ppo_ent_coef,
        clip_range=config.ppo_clip_range,
        policy_kwargs=policy_kwargs,
        use_sde=bool(getattr(config, "ppo_use_sde", False)),
        device=requested_device,
        seed=seed,
        verbose=0
    )

    callback = EpisodeRevenueCallback(target_episodes=config.train_episodes)
    pbar = tqdm(
        total=config.train_episodes,
        desc=f"PPO Seed {seed}",
        unit="ep",
        leave=False,
        disable=not show_progress,
    )
    while len(callback.episode_hotel_rewards) < config.train_episodes:
        prev_eps = len(callback.episode_hotel_rewards)
        model.learn(
            total_timesteps=config.ppo_n_steps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        curr_eps = len(callback.episode_hotel_rewards)
        pbar.update(max(0, curr_eps - prev_eps))
        pbar.set_postfix({"ep": len(callback.episode_hotel_rewards)})
    pbar.close()

    for idx, (hotel_rew, ota_profit) in enumerate(
        zip(callback.episode_hotel_rewards, callback.episode_ota_profits),
        start=1,
    ):
        train_records.append(
            {
                "Algorithm": algorithm_name,
                "Seed": seed,
                "Episode": idx,
                "EpisodeHotelRevenue": float(hotel_rew),
                "EpisodeOTAProfit": float(ota_profit),
                "EpisodeSystemProfit": float(hotel_rew + ota_profit),
                "EpisodeRevenue": float(hotel_rew),
            }
        )

    eval_rewards = evaluate_ppo_model(
        config=config,
        historical_data=historical_data,
        seed=seed + 400_000,
        model=model,
        vec_normalizer=vec_env,
        n_episodes=config.post_eval_episodes,
    )
    for idx, rew in enumerate(eval_rewards, start=1):
        eval_records.append(
            {
                "Algorithm": algorithm_name,
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalHotelRevenue": float(rew["EvalHotelRevenue"]),
                "EvalOTAProfit": float(rew["EvalOTAProfit"]),
                "EvalSystemProfit": float(rew["EvalSystemProfit"]),
                "EvalRevenue": float(rew["EvalHotelRevenue"]),
            }
        )
    vec_env.close()
    return train_records, eval_records, seed
