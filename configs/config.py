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
    train_years: tuple[int, ...] = (2016,)
    eval_years: tuple[int, ...] = (2017,)
    max_lead_time: int = 2
    adr_column: str = "adr"
    seed: int = 42


@dataclass
class ABMConfig:
    window_size: int = 3
    wtp_min: float = 30.0
    utility_noise_std: float = 8.0
    flexible_customer_share: float = 0.5
    lambda_day_mismatch_biz: float = 1000.0
    lambda_day_mismatch_flex: float = 12.0
    lambda_reference_price: float = 0.35
    reference_memory_alpha: float = 0.85
    weekday_arrival_fallback_mean: float = 18.0
    weekend_arrival_fallback_mean: float = 28.0


@dataclass
class EnvConfig:
    capacity: int = 50
    episode_days: int = 256
    price_min: float = 50.0
    price_max: float = 300.0
    full_capacity_penalty: float = 80.0
    penalty_scale_mode: str = "fixed"
    penalty_capacity_ref: int = 30
    scarcity_threshold_ratio: float = 0.3
    scarcity_penalty_coef: float = 400.0
    start_day: int = 0


@dataclass
class PPOConfig:
    total_timesteps: int = 800000
    learning_rate: float = 1e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 1  # 0.5
    max_grad_norm: float = 0.3
    target_kl: float = 0.02
    actor_net_arch: tuple[int, int] = (128, 128)
    critic_net_arch: tuple[int, int] = (256, 256, 256)
    seed: int = 42
    device: str = "auto"
    normalize_obs: bool = True
    normalize_reward: bool = True
    reward_clip: float = 10.0
    obs_clip: float = 10.0
    policy_variant: str = "tanh_gaussian"  # standard/tanh_gaussian/truncated_gaussian/scale_adjusted_truncated_gaussian/beta
    truncated_gaussian_k: float = 2.0
    truncated_gaussian_d_min: float = 0.01
    beta_min_concentration: float = 1.0
    save_name: str = "ppo_idea2_hotel"
    run_name: str = "idea2_ppo"
    log_interval: int = 10


@dataclass
class SACConfig:
    total_timesteps: int = 512000
    learning_rate: float = 3e-4
    buffer_size: int = 200000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    target_entropy: str = "auto"
    actor_net_arch: tuple[int, int] = (256, 256)
    critic_net_arch: tuple[int, int] = (256, 256)
    seed: int = 42
    device: str = "auto"
    normalize_obs: bool = True
    normalize_reward: bool = True
    reward_clip: float = 10.0
    obs_clip: float = 10.0
    save_name: str = "sac_idea2_hotel"
    run_name: str = "idea2_sac"
    log_interval: int = 10


@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    abm: ABMConfig = field(default_factory=ABMConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)

    def validate(self) -> None:
        if not self.data.train_years:
            raise ValueError("data.train_years 至少需要包含一个年份。")
        if not self.data.eval_years:
            raise ValueError("data.eval_years 至少需要包含一个年份。")
        if self.abm.window_size != 3:
            raise ValueError("idea2 模型要求滚动窗口固定为 3 天。")
        if not (0.0 <= self.abm.flexible_customer_share <= 1.0):
            raise ValueError("flexible_customer_share 必须位于 [0, 1] 内。")
        if self.abm.lambda_day_mismatch_biz < 0.0:
            raise ValueError("lambda_day_mismatch_biz 不能为负数。")
        if self.abm.lambda_day_mismatch_flex < 0.0:
            raise ValueError("lambda_day_mismatch_flex 不能为负数。")
        if self.env.price_min >= self.env.price_max:
            raise ValueError("price_min 必须小于 price_max。")
        if not (0.0 < self.abm.reference_memory_alpha < 1.0):
            raise ValueError("reference_memory_alpha 必须位于 (0, 1) 内。")
        if self.env.capacity <= 0:
            raise ValueError("capacity 必须为正数。")
        if self.env.episode_days <= 0:
            raise ValueError("episode_days 必须为正数。")
        if self.env.penalty_scale_mode not in {"fixed", "linear_capacity"}:
            raise ValueError("penalty_scale_mode 仅支持 'fixed' 或 'linear_capacity'。")
        if self.env.penalty_capacity_ref <= 0:
            raise ValueError("penalty_capacity_ref 必须为正数。")
        if not (0.0 < self.env.scarcity_threshold_ratio < 1.0):
            raise ValueError("scarcity_threshold_ratio 必须位于 (0, 1) 内。")
        if self.env.scarcity_penalty_coef < 0.0:
            raise ValueError("scarcity_penalty_coef 不能为负数。")
        if self.ppo.policy_variant not in {
            "standard",
            "tanh_gaussian",
            "truncated_gaussian",
            "scale_adjusted_truncated_gaussian",
            "beta",
        }:
            raise ValueError(
                "ppo.policy_variant 仅支持 'standard', 'tanh_gaussian', "
                "'truncated_gaussian', 'scale_adjusted_truncated_gaussian' 或 'beta'。"
            )
        if self.ppo.truncated_gaussian_k <= 0.0:
            raise ValueError("truncated_gaussian_k 必须为正数。")
        if not (0.0 < self.ppo.truncated_gaussian_d_min <= 1.0):
            raise ValueError("truncated_gaussian_d_min 必须位于 (0, 1] 内。")
        if self.ppo.beta_min_concentration <= 0.0:
            raise ValueError("beta_min_concentration 必须为正数。")
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
SAC_CONFIG = CONFIG.sac


def get_config() -> ProjectConfig:
    return CONFIG


__all__ = [
    "PROJECT_ROOT",
    "PathConfig",
    "DataConfig",
    "ABMConfig",
    "EnvConfig",
    "PPOConfig",
    "SACConfig",
    "ProjectConfig",
    "CONFIG",
    "PATH_CONFIG",
    "DATA_CONFIG",
    "ABM_CONFIG",
    "ENV_CONFIG",
    "PPO_CONFIG",
    "SAC_CONFIG",
    "get_config",
]
