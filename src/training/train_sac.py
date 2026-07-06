from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional

from configs.config import CONFIG, PATH_CONFIG, SAC_CONFIG
from src.environment.abm_customer_model import load_train_historical_data
from src.training.train_ppo import EpisodeMetricsAggregator, _apply_nested_overrides


def build_env(
    historical_data=None,
    seed: Optional[int] = None,
    capacity: Optional[int] = None,
    training: bool = True,
    norm_reward: Optional[bool] = None,
    env_overrides: Optional[dict[str, Any]] = None,
):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.environment.gym_hotel_env import GymHotelPricingEnv

    historical_data = load_train_historical_data() if historical_data is None else historical_data
    env_seed = int(SAC_CONFIG.seed if seed is None else seed)
    reward_norm = bool(SAC_CONFIG.normalize_reward if norm_reward is None else norm_reward)
    env_kwargs = dict(env_overrides or {})

    def _make_env():
        env = GymHotelPricingEnv(
            historical_data=historical_data,
            seed=env_seed,
            capacity=capacity,
            **env_kwargs,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize(
        vec_env,
        training=bool(training),
        norm_obs=bool(SAC_CONFIG.normalize_obs),
        norm_reward=reward_norm,
        clip_obs=float(SAC_CONFIG.obs_clip),
        clip_reward=float(SAC_CONFIG.reward_clip),
        gamma=float(SAC_CONFIG.gamma),
    )
    return vec_env


def save_run_artifacts(
    model,
    vec_env,
    run_dir: Path,
    config_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> None:
    model_path = run_dir / f"{SAC_CONFIG.save_name}.zip"
    norm_path = run_dir / f"{SAC_CONFIG.save_name}_vecnormalize.pkl"
    config_path = run_dir / "run_config.json"

    model.save(model_path)
    vec_env.save(str(norm_path))
    payload = {
        "paths": {k: str(v) for k, v in asdict(CONFIG.paths).items()},
        "data": asdict(CONFIG.data),
        "abm": asdict(CONFIG.abm),
        "env": asdict(CONFIG.env),
        "ppo": asdict(CONFIG.ppo),
        "sac": asdict(CONFIG.sac),
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
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未检测到 stable-baselines3。请先执行 `pip install -r requirements.txt`。"
        ) from exc

    return SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(SAC_CONFIG.learning_rate),
        buffer_size=int(SAC_CONFIG.buffer_size),
        learning_starts=int(SAC_CONFIG.learning_starts),
        batch_size=int(SAC_CONFIG.batch_size),
        tau=float(SAC_CONFIG.tau),
        gamma=float(SAC_CONFIG.gamma),
        train_freq=int(SAC_CONFIG.train_freq),
        gradient_steps=int(SAC_CONFIG.gradient_steps),
        ent_coef=SAC_CONFIG.ent_coef,
        target_entropy=SAC_CONFIG.target_entropy,
        tensorboard_log=str(PATH_CONFIG.tensorboard_dir if tensorboard_log is None else tensorboard_log),
        policy_kwargs={
            "net_arch": {
                "pi": list(SAC_CONFIG.actor_net_arch),
                "qf": list(SAC_CONFIG.critic_net_arch),
            }
        },
        seed=int(SAC_CONFIG.seed if seed is None else seed),
        device=str(SAC_CONFIG.device),
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
    env_overrides: Optional[dict[str, Any]] = None,
):
    effective_run_name = SAC_CONFIG.run_name if run_name is None else run_name
    effective_seed = int(SAC_CONFIG.seed if train_seed is None else train_seed)
    effective_timesteps = int(SAC_CONFIG.total_timesteps if total_timesteps is None else total_timesteps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PATH_CONFIG.model_dir / f"{effective_run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_env(
        historical_data=historical_data,
        seed=effective_seed,
        capacity=capacity,
        training=True,
        env_overrides=env_overrides,
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
        log_interval=int(SAC_CONFIG.log_interval),
        tb_log_name=effective_run_name,
        progress_bar=progress_bar,
    )
    env_config_overrides = {"capacity": int(capacity if capacity is not None else CONFIG.env.capacity)}
    if env_overrides:
        env_config_overrides.update(env_overrides)

    save_run_artifacts(
        model,
        vec_env,
        run_dir,
        config_overrides={
            "env": env_config_overrides,
            "sac": {
                "seed": effective_seed,
                "run_name": effective_run_name,
                "total_timesteps": effective_timesteps,
            },
        },
    )
    return model, vec_env, run_dir


def build_eval_env(
    train_vec_env,
    historical_data=None,
    seed: Optional[int] = None,
    capacity: Optional[int] = None,
    env_overrides: Optional[dict[str, Any]] = None,
):
    eval_env = build_env(
        historical_data=historical_data,
        seed=seed,
        capacity=capacity,
        training=False,
        norm_reward=False,
        env_overrides=env_overrides,
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

    historical_data = load_train_historical_data()
    _, vec_env, run_dir = train_single_run(
        run_name=SAC_CONFIG.run_name,
        historical_data=historical_data,
        capacity=CONFIG.env.capacity,
        train_seed=SAC_CONFIG.seed,
        total_timesteps=SAC_CONFIG.total_timesteps,
        progress_bar=True,
        verbose=1,
    )
    vec_env.close()
    print(f"训练完成，模型已保存到: {run_dir}")
    print(f"TensorBoard 日志目录: {PATH_CONFIG.tensorboard_dir}")


if __name__ == "__main__":
    main()
