from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from configs.config import CONFIG, PATH_CONFIG, PPO_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data


class EpisodeMetricsAggregator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.episode_revenue = 0.0
        self.episode_penalty = 0.0
        self.episode_full_penalty = 0.0
        self.episode_scarcity_penalty = 0.0
        self.episode_arrivals = 0.0
        self.episode_accepted = 0.0
        self.episode_prices: list[np.ndarray] = []
        self.episode_inventory: list[np.ndarray] = []
        self.episode_inventory_before: list[np.ndarray] = []
        self.episode_full_day_count = 0.0
        self.episode_full_slot_count = 0.0
        self.episode_full_slot_by_offset = np.zeros(3, dtype=float)
        self.episode_step_count = 0.0

    def update(self, info: dict[str, Any]) -> None:
        self.episode_revenue += float(info.get("revenue", 0.0))
        self.episode_full_penalty += float(info.get("full_penalty", 0.0))
        self.episode_scarcity_penalty += float(info.get("scarcity_penalty", 0.0))
        self.episode_penalty += float(info.get("total_penalty", info.get("full_penalty", 0.0)))
        self.episode_arrivals += float(info.get("arrivals", 0.0))
        accepted = np.asarray(info.get("accepted_by_offset", [0.0, 0.0, 0.0]), dtype=float)
        prices = np.asarray(info.get("prices", [0.0, 0.0, 0.0]), dtype=float)
        inventory_before = np.asarray(info.get("inventory_before", [0.0, 0.0, 0.0]), dtype=float)
        inventory = np.asarray(info.get("inventory_after", [0.0, 0.0, 0.0]), dtype=float)
        full_flags = (inventory <= 0.0).astype(float)

        self.episode_accepted += float(np.sum(accepted))
        self.episode_prices.append(prices)
        self.episode_inventory.append(inventory)
        self.episode_inventory_before.append(inventory_before)
        self.episode_full_day_count += float(np.any(full_flags > 0.0))
        self.episode_full_slot_count += float(np.sum(full_flags))
        self.episode_full_slot_by_offset += full_flags
        self.episode_step_count += 1.0

    def summary(self) -> dict[str, float]:
        stacked_prices = np.vstack(self.episode_prices) if self.episode_prices else np.zeros((1, 3))
        stacked_inventory = np.vstack(self.episode_inventory) if self.episode_inventory else np.zeros((1, 3))
        stacked_inventory_before = (
            np.vstack(self.episode_inventory_before) if self.episode_inventory_before else np.zeros((1, 3))
        )
        acceptance_rate = self.episode_accepted / max(1.0, self.episode_arrivals)
        full_day_rate = self.episode_full_day_count / max(1.0, self.episode_step_count)
        full_slot_rate = self.episode_full_slot_count / max(1.0, 3.0 * self.episode_step_count)
        full_slot_rate_by_offset = self.episode_full_slot_by_offset / max(1.0, self.episode_step_count)
        return {
            "episode_revenue": float(self.episode_revenue),
            "episode_penalty": float(self.episode_penalty),
            "episode_full_penalty": float(self.episode_full_penalty),
            "episode_scarcity_penalty": float(self.episode_scarcity_penalty),
            "episode_arrivals": float(self.episode_arrivals),
            "episode_accepted": float(self.episode_accepted),
            "episode_acceptance_rate": float(acceptance_rate),
            "avg_price": float(np.mean(stacked_prices)),
            "avg_inventory": float(np.mean(stacked_inventory)),
            "avg_inventory_before": float(np.mean(stacked_inventory_before)),
            "full_day_rate": float(full_day_rate),
            "full_slot_rate": float(full_slot_rate),
            "avg_price_day0": float(np.mean(stacked_prices[:, 0])),
            "avg_price_day1": float(np.mean(stacked_prices[:, 1])),
            "avg_price_day2": float(np.mean(stacked_prices[:, 2])),
            "avg_inventory_day0": float(np.mean(stacked_inventory[:, 0])),
            "avg_inventory_day1": float(np.mean(stacked_inventory[:, 1])),
            "avg_inventory_day2": float(np.mean(stacked_inventory[:, 2])),
            "avg_inventory_before_day0": float(np.mean(stacked_inventory_before[:, 0])),
            "avg_inventory_before_day1": float(np.mean(stacked_inventory_before[:, 1])),
            "avg_inventory_before_day2": float(np.mean(stacked_inventory_before[:, 2])),
            "full_rate_day0": float(full_slot_rate_by_offset[0]),
            "full_rate_day1": float(full_slot_rate_by_offset[1]),
            "full_rate_day2": float(full_slot_rate_by_offset[2]),
        }


def build_env(
    historical_data=None,
    seed: Optional[int] = None,
    capacity: Optional[int] = None,
    training: bool = True,
    norm_reward: Optional[bool] = None,
):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.environment.gym_hotel_env import GymHotelPricingEnv

    historical_data = load_filtered_historical_data() if historical_data is None else historical_data
    env_seed = int(PPO_CONFIG.seed if seed is None else seed)
    reward_norm = bool(PPO_CONFIG.normalize_reward if norm_reward is None else norm_reward)

    def _make_env():
        env = GymHotelPricingEnv(historical_data=historical_data, seed=env_seed, capacity=capacity)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize(
        vec_env,
        training=bool(training),
        norm_obs=bool(PPO_CONFIG.normalize_obs),
        norm_reward=reward_norm,
        clip_obs=float(PPO_CONFIG.obs_clip),
        clip_reward=float(PPO_CONFIG.reward_clip),
        gamma=float(PPO_CONFIG.gamma),
    )
    return vec_env


def _apply_nested_overrides(payload: dict[str, Any], overrides: Optional[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    if overrides is None:
        return payload
    for section, values in overrides.items():
        if section in payload and isinstance(payload[section], dict):
            payload[section].update(values)
        else:
            payload[section] = values
    return payload


def save_run_artifacts(
    model,
    vec_env,
    run_dir: Path,
    config_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> None:
    model_path = run_dir / f"{PPO_CONFIG.save_name}.zip"
    norm_path = run_dir / f"{PPO_CONFIG.save_name}_vecnormalize.pkl"
    config_path = run_dir / "run_config.json"

    model.save(model_path)
    vec_env.save(str(norm_path))
    payload = {
        "paths": {k: str(v) for k, v in asdict(CONFIG.paths).items()},
        "data": asdict(CONFIG.data),
        "abm": asdict(CONFIG.abm),
        "env": asdict(CONFIG.env),
        "ppo": asdict(CONFIG.ppo),
    }
    payload = _apply_nested_overrides(payload, config_overrides)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def create_tensorboard_callback():
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 stable-baselines3。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    class TensorboardMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.aggregator = EpisodeMetricsAggregator()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for idx, info in enumerate(infos):
                self.aggregator.update(info)
                done = bool(dones[idx]) if idx < len(dones) else False
                if done:
                    for key, value in self.aggregator.summary().items():
                        self.logger.record(f"custom/{key}", value)
                    self.aggregator.reset()
            return True

    return TensorboardMetricsCallback()


def create_model(vec_env, tensorboard_log: Optional[Path] = None, seed: Optional[int] = None, verbose: int = 1):
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 stable-baselines3。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(PPO_CONFIG.learning_rate),
        n_steps=int(PPO_CONFIG.n_steps),
        batch_size=int(PPO_CONFIG.batch_size),
        n_epochs=int(PPO_CONFIG.n_epochs),
        gamma=float(PPO_CONFIG.gamma),
        gae_lambda=float(PPO_CONFIG.gae_lambda),
        clip_range=float(PPO_CONFIG.clip_range),
        ent_coef=float(PPO_CONFIG.ent_coef),
        vf_coef=float(PPO_CONFIG.vf_coef),
        max_grad_norm=float(PPO_CONFIG.max_grad_norm),
        target_kl=float(PPO_CONFIG.target_kl),
        tensorboard_log=str(PATH_CONFIG.tensorboard_dir if tensorboard_log is None else tensorboard_log),
        policy_kwargs={
            "net_arch": {
                "pi": list(PPO_CONFIG.actor_net_arch),
                "vf": list(PPO_CONFIG.critic_net_arch),
            }
        },
        seed=int(PPO_CONFIG.seed if seed is None else seed),
        device=str(PPO_CONFIG.device),
        verbose=int(verbose),
    )


def train_single_run(
    run_name: Optional[str] = None,
    historical_data=None,
    capacity: Optional[int] = None,
    train_seed: Optional[int] = None,
    total_timesteps: Optional[int] = None,
    progress_bar: bool = True,
    verbose: int = 1,
):
    effective_run_name = PPO_CONFIG.run_name if run_name is None else run_name
    effective_seed = int(PPO_CONFIG.seed if train_seed is None else train_seed)
    effective_timesteps = int(PPO_CONFIG.total_timesteps if total_timesteps is None else total_timesteps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PATH_CONFIG.model_dir / f"{effective_run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_env(
        historical_data=historical_data,
        seed=effective_seed,
        capacity=capacity,
        training=True,
    )
    callback = create_tensorboard_callback()
    model = create_model(
        vec_env,
        tensorboard_log=PATH_CONFIG.tensorboard_dir,
        seed=effective_seed,
        verbose=verbose,
    )
    model.learn(
        total_timesteps=effective_timesteps,
        callback=callback,
        log_interval=int(PPO_CONFIG.log_interval),
        tb_log_name=effective_run_name,
        progress_bar=progress_bar,
    )
    save_run_artifacts(
        model,
        vec_env,
        run_dir,
        config_overrides={
            "env": {"capacity": int(capacity if capacity is not None else CONFIG.env.capacity)},
            "ppo": {
                "seed": effective_seed,
                "run_name": effective_run_name,
                "total_timesteps": effective_timesteps,
            },
        },
    )
    return model, vec_env, run_dir


def build_eval_env(train_vec_env, historical_data=None, seed: Optional[int] = None, capacity: Optional[int] = None):
    eval_env = build_env(
        historical_data=historical_data,
        seed=seed,
        capacity=capacity,
        training=False,
        norm_reward=False,
    )
    eval_env.obs_rms = deepcopy(train_vec_env.obs_rms)
    if hasattr(train_vec_env, "ret_rms"):
        eval_env.ret_rms = deepcopy(train_vec_env.ret_rms)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def main() -> None:
    try:
        __import__("stable_baselines3")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 stable-baselines3。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    historical_data = load_filtered_historical_data()
    _, vec_env, run_dir = train_single_run(
        run_name=PPO_CONFIG.run_name,
        historical_data=historical_data,
        capacity=CONFIG.env.capacity,
        train_seed=PPO_CONFIG.seed,
        total_timesteps=PPO_CONFIG.total_timesteps,
        progress_bar=True,
        verbose=1,
    )
    vec_env.close()
    print(f"训练完成，模型已保存到: {run_dir}")
    print(f"TensorBoard 日志目录: {PATH_CONFIG.tensorboard_dir}")


if __name__ == "__main__":
    main()
