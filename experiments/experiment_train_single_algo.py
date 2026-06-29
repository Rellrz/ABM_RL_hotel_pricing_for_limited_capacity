from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIG, PATH_CONFIG
from src.environment.abm_customer_model import load_filtered_historical_data
from src.training.algorithm_registry import get_algorithm_choices, get_algorithm_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单次训练实验入口")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=get_algorithm_choices(),
        help="训练算法",
    )
    parser.add_argument("--train-seed", type=int, default=None, help="训练随机种子，默认使用所选算法配置")
    parser.add_argument("--total-timesteps", type=int, default=None, help="训练步数，默认使用所选算法配置")
    parser.add_argument("--run-name", type=str, default=None, help="可选的 run_name 覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = get_algorithm_runner(args.algo)
    algo_config = runner["config"]
    train_single_run = runner["train_single_run"]
    historical_data = load_filtered_historical_data()
    _, vec_env, run_dir = train_single_run(
        run_name=str(algo_config.run_name if args.run_name is None else args.run_name),
        historical_data=historical_data,
        capacity=CONFIG.env.capacity,
        train_seed=int(algo_config.seed if args.train_seed is None else args.train_seed),
        total_timesteps=int(algo_config.total_timesteps if args.total_timesteps is None else args.total_timesteps),
        progress_bar=True,
        verbose=1,
    )
    vec_env.close()
    print(f"训练完成，模型已保存到: {run_dir}")
    print(f"TensorBoard 日志目录: {PATH_CONFIG.tensorboard_dir}")


if __name__ == "__main__":
    main()
