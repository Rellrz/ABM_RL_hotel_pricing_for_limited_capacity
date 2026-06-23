from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_path: Path = PROJECT_ROOT / "datasets" / "hotel_bookings.csv"
    output_root: Path = PROJECT_ROOT / "outputs"
    model_dir: Path = PROJECT_ROOT / "outputs" / "models"
    tensorboard_dir: Path = PROJECT_ROOT / "outputs" / "tensorboard"
    log_dir: Path = PROJECT_ROOT / "outputs" / "logs"

    def ensure_dirs(self) -> None:
        for path in (self.output_root, self.model_dir, self.tensorboard_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    hotel_name: str = "City Hotel"
    max_lead_time: int = 2
    adr_column: str = "adr"
    seed: int = 42


@dataclass
class ABMConfig:
    window_size: int = 3
    wtp_min: float = 30.0
    utility_noise_std: float = 8.0
    lambda_day_mismatch: float = 12.0
    lambda_reference_price: float = 0.35
    reference_memory_alpha: float = 0.85
    weekday_arrival_fallback_mean: float = 18.0
    weekend_arrival_fallback_mean: float = 28.0


@dataclass
class EnvConfig:
    capacity: int = 30
    episode_days: int = 400
    price_min: float = 50.0
    price_max: float = 300.0
    full_capacity_penalty: float = 80.0
    start_day: int = 0


@dataclass
class PPOConfig:
    total_timesteps: int = 800_000
    learning_rate: float = 1e-4
    n_steps: int = 128
    batch_size: int = 40
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.3
    target_kl: float = 0.02
    actor_net_arch: tuple[int, int] = (128, 128)
    critic_net_arch: tuple[int, int] = (256, 256)
    seed: int = 42
    device: str = "auto"
    normalize_obs: bool = True
    normalize_reward: bool = True
    reward_clip: float = 10.0
    obs_clip: float = 10.0
    save_name: str = "ppo_idea2_hotel"
    run_name: str = "idea2_ppo"
    log_interval: int = 10


@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    abm: ABMConfig = field(default_factory=ABMConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self) -> None:
        if self.abm.window_size != 3:
            raise ValueError("idea2 模型要求滚动窗口固定为 3 天。")
        if self.env.price_min >= self.env.price_max:
            raise ValueError("price_min 必须小于 price_max。")
        if not (0.0 < self.abm.reference_memory_alpha < 1.0):
            raise ValueError("reference_memory_alpha 必须位于 (0, 1) 内。")
        if self.env.capacity <= 0:
            raise ValueError("capacity 必须为正数。")
        if self.env.episode_days <= 0:
            raise ValueError("episode_days 必须为正数。")

    def setup(self) -> None:
        self.validate()
        self.paths.ensure_dirs()


CONFIG = ProjectConfig()
CONFIG.setup()

PATH_CONFIG = CONFIG.paths
DATA_CONFIG = CONFIG.data
ABM_CONFIG = CONFIG.abm
ENV_CONFIG = CONFIG.env
PPO_CONFIG = CONFIG.ppo


def get_config() -> ProjectConfig:
    return CONFIG


__all__ = [
    "PROJECT_ROOT",
    "PathConfig",
    "DataConfig",
    "ABMConfig",
    "EnvConfig",
    "PPOConfig",
    "ProjectConfig",
    "CONFIG",
    "PATH_CONFIG",
    "DATA_CONFIG",
    "ABM_CONFIG",
    "ENV_CONFIG",
    "PPO_CONFIG",
    "get_config",
]
