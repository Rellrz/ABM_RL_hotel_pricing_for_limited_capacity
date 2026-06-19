"""实验一配置模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = THIS_DIR / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"


@dataclass
class ExperimentConfig:
    """集中管理实验一的全部参数，避免魔法数字散落在代码中。"""

    # -----------------------------
    # 环境参数
    # -----------------------------
    max_inv: int = 17
    max_lt: int = 7
    price_min: float = 100.0
    price_max: float = 200.0
    commission_rate: float = 0.15
    base_arrival_rate: float = 5.0
    price_sensitivity: float = 0.02

    # -----------------------------
    # MDP 求解参数
    # -----------------------------
    grid_size: int = 10
    n_mc_samples: int = 10000
    gamma: float = 0.99
    vi_tol: float = 1e-4
    vi_max_iter: int = 10000

    # -----------------------------
    # CEM 参数
    # -----------------------------
    n_episodes: int = 300
    n_rollouts_per_episode: int = 100
    elite_frac: float = 0.3
    alpha: float = 0.45
    init_mean_online: float = 150.0
    init_mean_offline: float = 150.0
    init_var_online: float = 400.0
    init_var_offline: float = 400.0
    cov_reg: float = 1e-6
    min_std: float = 2.0
    std_decay: float = 0.99
    memory_size: int = 400
    min_update_samples: int = 20
    deterministic_eval: bool = True
    neighbor_smoothing: float = 0.20

    # -----------------------------
    # 实验执行参数
    # -----------------------------
    n_seeds: int = 30
    static_eval_rollouts: int = 50
    static_search_seed: int = 20260421
    base_seed: int = 20260421
    n_jobs: int = max(1, min((os.cpu_count() or 1), 30))

    # -----------------------------
    # 路径参数
    # -----------------------------
    cache_path: Path = CACHE_DIR / "mdp_cache.pkl"
    results_csv_path: Path = RESULTS_DIR / "experiment1_results.csv"
    summary_path: Path = RESULTS_DIR / "experiment1_summary.json"
    figure_path: Path = FIGURES_DIR / "convergence.pdf"

    def ensure_directories(self) -> None:
        """确保实验输出目录存在。"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def n_inventory_states(self) -> int:
        return self.max_inv + 1

    @property
    def n_lead_time_states(self) -> int:
        return self.max_lt + 1

    @property
    def init_mean(self) -> np.ndarray:
        return np.array([self.init_mean_online, self.init_mean_offline], dtype=float)

    @property
    def init_cov(self) -> np.ndarray:
        return np.diag([self.init_var_online, self.init_var_offline]).astype(float)


DEFAULT_CONFIG = ExperimentConfig()
