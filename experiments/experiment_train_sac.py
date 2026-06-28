from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIG, PATH_CONFIG, SAC_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data
from src.training.train_sac import train_single_run


def main() -> None:
    historical_data = load_filtered_historical_data()
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
