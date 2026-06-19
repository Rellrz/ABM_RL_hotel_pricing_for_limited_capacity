from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from configs.config import CONFIG, PATH_CONFIG, PPO_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data


def build_env():
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.environment.gym_hotel_env import GymHotelPricingEnv

    historical_data = load_filtered_historical_data()

    def _make_env():
        env = GymHotelPricingEnv(historical_data=historical_data, seed=PPO_CONFIG.seed)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=bool(PPO_CONFIG.normalize_obs),
        norm_reward=bool(PPO_CONFIG.normalize_reward),
        clip_obs=float(PPO_CONFIG.obs_clip),
        clip_reward=float(PPO_CONFIG.reward_clip),
        gamma=float(PPO_CONFIG.gamma),
    )
    return vec_env


def save_run_artifacts(model, vec_env, run_dir: Path) -> None:
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
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 stable-baselines3。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    class TensorboardMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_revenue = 0.0
            self.episode_penalty = 0.0
            self.episode_arrivals = 0.0
            self.episode_accepted = 0.0
            self.episode_prices: list[np.ndarray] = []
            self.episode_inventory: list[np.ndarray] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for idx, info in enumerate(infos):
                self.episode_revenue += float(info.get("revenue", 0.0))
                self.episode_penalty += float(info.get("full_penalty", 0.0))
                self.episode_arrivals += float(info.get("arrivals", 0.0))
                accepted = np.asarray(info.get("accepted_by_offset", [0.0, 0.0, 0.0]), dtype=float)
                prices = np.asarray(info.get("prices", [0.0, 0.0, 0.0]), dtype=float)
                inventory = np.asarray(info.get("inventory_after", [0.0, 0.0, 0.0]), dtype=float)
                self.episode_accepted += float(np.sum(accepted))
                self.episode_prices.append(prices)
                self.episode_inventory.append(inventory)

                done = bool(dones[idx]) if idx < len(dones) else False
                if done:
                    avg_price = float(np.mean(np.vstack(self.episode_prices))) if self.episode_prices else 0.0
                    avg_inventory = float(np.mean(np.vstack(self.episode_inventory))) if self.episode_inventory else 0.0
                    acceptance_rate = self.episode_accepted / max(1.0, self.episode_arrivals)
                    self.logger.record("custom/episode_revenue", self.episode_revenue)
                    self.logger.record("custom/episode_penalty", self.episode_penalty)
                    self.logger.record("custom/episode_arrivals", self.episode_arrivals)
                    self.logger.record("custom/episode_accepted", self.episode_accepted)
                    self.logger.record("custom/episode_acceptance_rate", acceptance_rate)
                    self.logger.record("custom/avg_price", avg_price)
                    self.logger.record("custom/avg_inventory", avg_inventory)
                    self.episode_revenue = 0.0
                    self.episode_penalty = 0.0
                    self.episode_arrivals = 0.0
                    self.episode_accepted = 0.0
                    self.episode_prices = []
                    self.episode_inventory = []
            return True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PATH_CONFIG.model_dir / f"{PPO_CONFIG.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_env()
    callback = TensorboardMetricsCallback()
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(PPO_CONFIG.learning_rate),
        n_steps=int(PPO_CONFIG.n_steps),
        batch_size=int(PPO_CONFIG.batch_size),
        gamma=float(PPO_CONFIG.gamma),
        gae_lambda=float(PPO_CONFIG.gae_lambda),
        clip_range=float(PPO_CONFIG.clip_range),
        ent_coef=float(PPO_CONFIG.ent_coef),
        vf_coef=float(PPO_CONFIG.vf_coef),
        max_grad_norm=float(PPO_CONFIG.max_grad_norm),
        tensorboard_log=str(PATH_CONFIG.tensorboard_dir),
        policy_kwargs={"net_arch": list(PPO_CONFIG.net_arch)},
        seed=int(PPO_CONFIG.seed),
        device=str(PPO_CONFIG.device),
        verbose=1,
    )
    model.learn(
        total_timesteps=int(PPO_CONFIG.total_timesteps),
        callback=callback,
        log_interval=int(PPO_CONFIG.log_interval),
        tb_log_name=PPO_CONFIG.run_name,
        progress_bar=True,
    )
    save_run_artifacts(model, vec_env, run_dir)
    vec_env.close()
    print(f"训练完成，模型已保存到: {run_dir}")
    print(f"TensorBoard 日志目录: {PATH_CONFIG.tensorboard_dir}")


if __name__ == "__main__":
    main()
