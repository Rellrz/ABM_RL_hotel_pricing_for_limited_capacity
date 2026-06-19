"""实验二配置：对比与消融实验。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ENV_CONFIG, RL_CONFIG


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "outputs" / "experiment2"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"
TUNING_DIR = ARTIFACTS_DIR / "tuning"
TUNING_FIGURES_DIR = TUNING_DIR / "figures"


DEFAULT_BUCKET_SPEC = "0|1|2-3|4-6|7-13|14-29|30-59|60-90"


@dataclass
class Experiment2Config:
    # -----------------------------
    # 运行规模（可切换）
    # -----------------------------
    run_mode: str = "debug"  # debug / medium / full
    post_eval_episodes: int = 30
    override_train_episodes: int | None = None
    days_per_episode: int = 730
    n_jobs: int = 1

    # -----------------------------
    # 环境与业务参数（与现有项目对齐）
    # -----------------------------
    initial_inventory: int = int(ENV_CONFIG.initial_inventory)
    booking_window_days: int = int(ENV_CONFIG.booking_window_days)
    commission_rate: float = float(RL_CONFIG.commission_rate)
    reward_hotel_ratio: float = float(RL_CONFIG.reward_hotel_ratio)
    reward_shape_price_weight: float = float(RL_CONFIG.reward_shape_price_weight)
    reward_shape_sellthrough_weight: float = float(RL_CONFIG.reward_shape_sellthrough_weight)
    reward_shape_target_sellthrough: float = float(RL_CONFIG.reward_shape_target_sellthrough)
    online_price_min: float = float(RL_CONFIG.online_price_min)
    online_price_max: float = float(RL_CONFIG.online_price_max)
    offline_price_min: float = float(RL_CONFIG.offline_price_min)
    offline_price_max: float = float(RL_CONFIG.offline_price_max)
    decision_buckets: str = DEFAULT_BUCKET_SPEC
    update_frequency: int = int(RL_CONFIG.update_frequency)
    ota_r_max: float = float(RL_CONFIG.subsidy_ratio_max)
    ota_delta_max: float = float(RL_CONFIG.ota_delta_max)
    ota_decay_lambda: float = float(RL_CONFIG.ota_decay_lambda)
    ota_noise_std: float = float(RL_CONFIG.ota_noise_std)
    ota_seed: int = int(RL_CONFIG.ota_seed)

    # -----------------------------
    # 离散化动作（Q-learning）
    # -----------------------------
    q_grid_size: int = 10
    q_alpha: float = 0.1
    q_gamma: float = 0.99
    q_eps_start: float = 1.0
    q_eps_end: float = 0.05
    q_eps_decay_steps: int = 300_000

    # -----------------------------
    # PPO参数
    # -----------------------------
    ppo_learning_rate: float = 1e-4
    ppo_n_steps: int | None = 256
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.995
    ppo_gae_lambda: float = 0.98
    ppo_ent_coef: float = 0.005
    ppo_clip_range: float = 0.2
    ppo_norm_obs: bool = True
    ppo_norm_reward: bool = False
    ppo_clip_obs: float = 10.0
    ppo_clip_reward: float = 10.0
    ppo_reward_mode: str = "mixed"
    ppo_reward_scale: float = 1e4
    ppo_shaped_reward_weight: float = 0.5
    ppo_net_arch: tuple = (256, 256)
    ppo_use_sde: bool = False
    ppo_device: str = "mps"
    ppo_log_std_init: float = -1.0

    # -----------------------------
    # CEM参数（复用项目配置）
    # -----------------------------
    cem_n_samples: int = int(RL_CONFIG.cem_n_samples)
    cem_elite_frac: float = float(RL_CONFIG.cem_elite_frac)
    cem_initial_std: float = float(RL_CONFIG.initial_std)
    cem_min_std: float = float(RL_CONFIG.min_std)
    cem_std_decay: float = float(RL_CONFIG.std_decay)
    cem_memory_size: int = int(RL_CONFIG.cem_memory_size)

    # -----------------------------
    # Bayesian Optimization 参数
    # -----------------------------
    bo_n_calls: int = 500
    bo_n_initial_points: int = 20
    bo_acq_func: str = "EI"
    bo_n_eval_episodes_per_point: int = 1

    # -----------------------------
    # Genetic Algorithm 参数
    # -----------------------------
    ga_pop_size: int = 40
    ga_n_generations: int = 25
    ga_tournament_pressure: int = 3
    ga_crossover_prob: float = 0.9
    ga_crossover_eta: float = 15.0
    ga_mutation_eta: float = 20.0

    # -----------------------------
    # Simulated Annealing 参数
    # -----------------------------
    sa_maxfun: int = 1000
    sa_initial_temp: float = 5230.0
    sa_visit: float = 2.62
    sa_accept: float = -5.0
    sa_no_local_search: bool = True

    # -----------------------------
    # Random Search 参数
    # -----------------------------
    rs_n_iterations: int = 1000

    # -----------------------------
    # 路径
    # -----------------------------
    run_timestamp: str = ""
    training_csv_path: Path = RESULTS_DIR / "experiment2_training.csv"
    evaluation_csv_path: Path = RESULTS_DIR / "experiment2_post_eval.csv"
    summary_json_path: Path = RESULTS_DIR / "experiment2_summary.json"
    stats_hotel_csv_path: Path = RESULTS_DIR / "experiment2_stats_hotel.csv"
    stats_ota_csv_path: Path = RESULTS_DIR / "experiment2_stats_ota.csv"
    stats_system_csv_path: Path = RESULTS_DIR / "experiment2_stats_system.csv"
    learning_curve_hotel_pdf: Path = FIGURES_DIR / "episode_revenue_curves_hotel.pdf"
    learning_curve_ota_pdf: Path = FIGURES_DIR / "episode_profit_curves_ota.pdf"
    learning_curve_system_pdf: Path = FIGURES_DIR / "episode_total_profit_curves_system.pdf"
    eval_bar_hotel_pdf: Path = FIGURES_DIR / "post_eval_bar_hotel_with_errorbars.pdf"
    eval_bar_ota_pdf: Path = FIGURES_DIR / "post_eval_bar_ota_with_errorbars.pdf"
    eval_bar_system_pdf: Path = FIGURES_DIR / "post_eval_bar_system_with_errorbars.pdf"
    performance_table_hotel_csv: Path = RESULTS_DIR / "performance_table_hotel.csv"
    performance_table_ota_csv: Path = RESULTS_DIR / "performance_table_ota.csv"
    performance_table_system_csv: Path = RESULTS_DIR / "performance_table_system.csv"
    tuning_trials_csv_path: Path = TUNING_DIR / "ppo_trials.csv"
    tuning_train_csv_path: Path = TUNING_DIR / "ppo_trial_training.csv"
    tuning_eval_csv_path: Path = TUNING_DIR / "ppo_trial_eval.csv"
    tuning_best_json_path: Path = TUNING_DIR / "ppo_best_config.json"
    tuning_summary_json_path: Path = TUNING_DIR / "ppo_tuning_summary.json"

    def __post_init__(self) -> None:
        if not self.run_timestamp:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.learning_curve_hotel_pdf = self._timestamped_figure_path(self.learning_curve_hotel_pdf)
        self.learning_curve_ota_pdf = self._timestamped_figure_path(self.learning_curve_ota_pdf)
        self.learning_curve_system_pdf = self._timestamped_figure_path(self.learning_curve_system_pdf)
        self.eval_bar_hotel_pdf = self._timestamped_figure_path(self.eval_bar_hotel_pdf)
        self.eval_bar_ota_pdf = self._timestamped_figure_path(self.eval_bar_ota_pdf)
        self.eval_bar_system_pdf = self._timestamped_figure_path(self.eval_bar_system_pdf)
        # 默认让PPO rollout长度与episode长度一致，避免跨年episode被中途切成多段更新。
        if self.ppo_n_steps is None:
            self.ppo_n_steps = int(self.days_per_episode)

    def _timestamped_figure_path(self, path: Path) -> Path:
        return path.with_name(f"{path.stem}_{self.run_timestamp}{path.suffix}")

    def ensure_dirs(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        TUNING_DIR.mkdir(parents=True, exist_ok=True)
        TUNING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def mode_profile(self) -> Dict[str, int]:
        profiles = {
            "debug": {"n_seeds": 1, "train_episodes": 300},
            "medium": {"n_seeds": 5, "train_episodes": 1000},
            "full": {"n_seeds": 30, "train_episodes": 1000},
        }
        if self.run_mode not in profiles:
            raise ValueError(f"Unknown run_mode={self.run_mode}")
        return profiles[self.run_mode]

    @property
    def n_seeds(self) -> int:
        return int(self.mode_profile["n_seeds"])

    @property
    def train_episodes(self) -> int:
        if self.override_train_episodes is not None:
            return int(self.override_train_episodes)
        return int(self.mode_profile["train_episodes"])

    @property
    def train_steps(self) -> int:
        return int(self.train_episodes * self.days_per_episode)

    @property
    def seed_list(self) -> List[int]:
        return list(range(1, self.n_seeds + 1))

    @property
    def n_stages(self) -> int:
        # 默认8个分桶，和现有配置一致
        return 8

    @property
    def q_n_states(self) -> int:
        # 复用 CEM 状态空间: stage(8) × season(3) × weekday(2) × near_inv(5) × far_inv(5) = 1200
        return 1200

    @property
    def q_action_grid(self) -> np.ndarray:
        points_on = np.linspace(self.online_price_min, self.online_price_max, self.q_grid_size)
        points_off = np.linspace(self.offline_price_min, self.offline_price_max, self.q_grid_size)
        actions = []
        for pon in points_on:
            for poff in points_off:
                actions.append([float(pon), float(poff)])
        return np.asarray(actions, dtype=np.float64)
