#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
conda run -n abm python experiments/compare_cem_init_strategies.py \
  --episodes 400 \
  --tail-window 50 \
  --blend-alpha 0.7 \
  --env-seed 42
'''
"""比较 CEM 不同初始均值策略：midpoint / emsrb / blended。"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import PATH_CONFIG, RANDOM_CONFIG, RL_CONFIG
from src.training.game_trainer import train_game_system


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较 CEM 初始均值策略")
    parser.add_argument("--data", type=str, default="datasets/hotel_bookings.csv", help="酒店预订数据文件路径")
    parser.add_argument("--episodes", type=int, default=400, help="训练轮数")
    parser.add_argument("--episode-days", type=int, default=730, help="每个 episode 的模拟天数")
    parser.add_argument("--mode", type=str, default="simultaneous", choices=["fixed_ota", "alternating", "simultaneous"], help="训练模式")
    parser.add_argument("--commission", type=float, default=0.20, help="OTA佣金率")
    parser.add_argument("--subsidy-ratio-max", type=float, default=0.8, help="最大补贴比例")
    parser.add_argument("--update-frequency", type=int, default=30, help="CEM 参数更新频率")
    parser.add_argument("--booking-window-days", type=int, default=91, help="预订窗口长度")
    parser.add_argument("--decision-buckets", type=str, default="0|1|2-3|4-6|7-13|14-29|30-59|60-90", help="提前期分桶")
    parser.add_argument("--blend-alpha", type=float, default=0.7, help="blended 策略中 EMSR-b prior 权重")
    parser.add_argument("--tail-window", type=int, default=50, help="收敛段统计窗口长度")
    parser.add_argument("--env-seed", type=int, default=42, help="固定环境随机种子，保证三组对比公平")
    parser.add_argument("--n-jobs", type=int, default=3, help="并行进程数；默认三种策略同时跑")
    return parser


def _tail_mean(series: pd.Series, window: int) -> float:
    n = max(1, min(len(series), int(window)))
    return float(series.tail(n).mean())


def _tail_std(series: pd.Series, window: int) -> float:
    n = max(1, min(len(series), int(window)))
    return float(series.tail(n).std(ddof=0))


def _run_single_strategy(
    strategy: str,
    args_dict: dict,
) -> tuple[dict, pd.DataFrame]:
    historical_data = pd.read_csv(args_dict["data"])
    historical_data = historical_data[historical_data["hotel"] == "City Hotel"].copy()

    old_random_mode = RANDOM_CONFIG.random_mode
    old_fixed_seed = RANDOM_CONFIG.fixed_seed
    old_commission = RL_CONFIG.commission_rate
    old_subsidy_ratio_max = RL_CONFIG.subsidy_ratio_max
    old_init_strategy = RL_CONFIG.cem_init_strategy
    old_blend_alpha = RL_CONFIG.cem_init_blend_alpha

    try:
        RANDOM_CONFIG.random_mode = "fixed"
        RANDOM_CONFIG.fixed_seed = int(args_dict["env_seed"])
        RL_CONFIG.commission_rate = float(args_dict["commission"])
        RL_CONFIG.subsidy_ratio_max = float(args_dict["subsidy_ratio_max"])
        RL_CONFIG.cem_init_strategy = str(strategy)
        RL_CONFIG.cem_init_blend_alpha = float(args_dict["blend_alpha"])
        np.random.seed(int(args_dict["env_seed"]))

        _, _, _, _, episode_info = train_game_system(
            historical_data=historical_data,
            episodes=int(args_dict["episodes"]),
            training_mode=str(args_dict["mode"]),
            update_frequency=int(args_dict["update_frequency"]),
            booking_window_days=int(args_dict["booking_window_days"]),
            decision_buckets=str(args_dict["decision_buckets"]),
            episode_days=int(args_dict["episode_days"]),
        )

        df = pd.DataFrame(episode_info)
        df["init_strategy"] = str(strategy)
        summary_row = {
            "init_strategy": str(strategy),
            "episodes": int(args_dict["episodes"]),
            "tail_window": int(args_dict["tail_window"]),
            "hotel_last": float(df["hotel_revenue"].iloc[-1]),
            "hotel_best": float(df["hotel_revenue"].max()),
            "hotel_tail_mean": _tail_mean(df["hotel_revenue"], int(args_dict["tail_window"])),
            "hotel_tail_std": _tail_std(df["hotel_revenue"], int(args_dict["tail_window"])),
            "ota_tail_mean": _tail_mean(df["ota_profit"], int(args_dict["tail_window"])),
            "system_tail_mean": _tail_mean(df["hotel_revenue"] + df["ota_profit"], int(args_dict["tail_window"])),
            "online_tail_mean": _tail_mean(df["bookings_online"], int(args_dict["tail_window"])),
            "offline_tail_mean": _tail_mean(df["bookings_offline"], int(args_dict["tail_window"])),
            "train_base_tail_mean": _tail_mean(df["train_base_reward_hotel"], int(args_dict["tail_window"])),
            "train_shaped_tail_mean": _tail_mean(df["train_shaped_reward_hotel"], int(args_dict["tail_window"])),
            "shape_penalty_tail_mean": _tail_mean(df["avg_shaping_penalty"], int(args_dict["tail_window"])),
        }
        return summary_row, df
    finally:
        RANDOM_CONFIG.random_mode = old_random_mode
        RANDOM_CONFIG.fixed_seed = old_fixed_seed
        RL_CONFIG.commission_rate = old_commission
        RL_CONFIG.subsidy_ratio_max = old_subsidy_ratio_max
        RL_CONFIG.cem_init_strategy = old_init_strategy
        RL_CONFIG.cem_init_blend_alpha = old_blend_alpha


def main() -> None:
    args = build_parser().parse_args()

    strategies = ["midpoint", "emsrb", "blended"]
    summary_rows: list[dict] = []
    episode_frames: list[pd.DataFrame] = []

    print("=" * 72)
    print("CEM 初始均值策略对比实验")
    print("=" * 72)
    print(f"环境种子: {args.env_seed}")
    print(f"训练轮数: {args.episodes}")
    print(f"比较策略: {strategies}")
    print(f"并行进程数: {max(1, min(int(args.n_jobs), len(strategies)))}")

    args_dict = vars(args).copy()
    max_workers = max(1, min(int(args.n_jobs), len(strategies)))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_single_strategy, strategy, args_dict): strategy
            for strategy in strategies
        }
        for fut in as_completed(futures):
            strategy = futures[fut]
            print("\n" + "-" * 72)
            print(f"完成策略: {strategy}")
            print("-" * 72)
            summary_row, df = fut.result()
            summary_rows.append(summary_row)
            episode_frames.append(df)

    summary_df = pd.DataFrame(summary_rows).sort_values("hotel_tail_mean", ascending=False).reset_index(drop=True)
    episode_df = pd.concat(episode_frames, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(PATH_CONFIG.results_dir, f"cem_init_compare_summary_{timestamp}.csv")
    detail_path = os.path.join(PATH_CONFIG.results_dir, f"cem_init_compare_episodes_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    episode_df.to_csv(detail_path, index=False)

    print("\n" + "=" * 72)
    print("实验完成")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"汇总结果已保存到: {summary_path}")
    print(f"逐轮结果已保存到: {detail_path}")


if __name__ == "__main__":
    main()
