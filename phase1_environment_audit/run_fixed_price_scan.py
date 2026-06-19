from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_utils import FIXED_PRICE_SCAN_DIR, ensure_phase1_dirs, load_city_hotel_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment2 import Experiment2Config
from src.environment.bucket_pricing_simulator import BucketPricingSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第一阶段：固定价格扫描")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--price-min", type=float, default=60.0)
    parser.add_argument("--price-max", type=float, default=160.0)
    parser.add_argument("--price-step", type=float, default=10.0)
    parser.add_argument("--offline-premium", type=float, default=0.0)
    parser.add_argument("--n-seeds", type=int, default=None)
    return parser


def iter_prices(price_min: float, price_max: float, price_step: float) -> list[float]:
    if price_step <= 0:
        raise ValueError("price_step must be positive")
    values = np.arange(price_min, price_max + 1e-9, price_step, dtype=float)
    return [float(v) for v in values]


def run_single_episode(config: Experiment2Config, historical_data, seed: int, pon: float, poff: float) -> dict:
    sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
    sim.reset()

    total_hotel = 0.0
    total_ota = 0.0
    total_online_bookings = 0
    total_offline_bookings = 0
    done = False

    while not done:
        stage_actions = [(pon, poff)] * sim.n_stages
        out = sim.step_day(stage_actions)
        total_hotel += out.reward_hotel
        total_ota += out.reward_ota
        for bucket in out.info.get("bookings_by_day_offset", []):
            total_online_bookings += int(bucket.get("bookings_online", 0))
            total_offline_bookings += int(bucket.get("bookings_offline", 0))
        done = out.done

    env_stats = sim.env.get_statistics()
    total_bookings = total_online_bookings + total_offline_bookings
    return {
        "HotelRevenue": float(total_hotel),
        "OTAProfit": float(total_ota),
        "SystemProfit": float(total_hotel + total_ota),
        "OnlineBookings": int(total_online_bookings),
        "OfflineBookings": int(total_offline_bookings),
        "TotalBookings": int(total_bookings),
        "OnlineShare": float(total_online_bookings / total_bookings) if total_bookings > 0 else 0.0,
        "AverageOccupancyProxy": float(env_stats.get("average_occupancy_rate", 0.0)),
        "AverageDailyRevenue": float(env_stats.get("average_daily_revenue", 0.0)),
    }


def main() -> None:
    args = build_parser().parse_args()
    ensure_phase1_dirs()

    config = Experiment2Config(run_mode=args.mode)
    historical_data = load_city_hotel_data()
    prices = iter_prices(args.price_min, args.price_max, args.price_step)
    seeds = config.seed_list if args.n_seeds is None else config.seed_list[: int(args.n_seeds)]

    episode_records = []
    for price in prices:
        pon = float(np.clip(price, config.online_price_min, config.online_price_max))
        poff = float(np.clip(price + args.offline_premium, config.offline_price_min, config.offline_price_max))
        for seed in seeds:
            for ep in range(args.n_episodes):
                run_seed = seed * 1000 + ep
                metrics = run_single_episode(config, historical_data, run_seed, pon, poff)
                episode_records.append(
                    {
                        "FixedOnlinePrice": float(pon),
                        "FixedOfflinePrice": float(poff),
                        "Seed": int(seed),
                        "EvalEpisode": int(ep + 1),
                        **metrics,
                    }
                )

    episode_df = pd.DataFrame(episode_records)
    summary_df = (
        episode_df.groupby(["FixedOnlinePrice", "FixedOfflinePrice"], as_index=False)
        .agg(
            HotelRevenueMean=("HotelRevenue", "mean"),
            HotelRevenueStd=("HotelRevenue", "std"),
            OTAProfitMean=("OTAProfit", "mean"),
            SystemProfitMean=("SystemProfit", "mean"),
            OnlineShareMean=("OnlineShare", "mean"),
            AverageOccupancyProxyMean=("AverageOccupancyProxy", "mean"),
            AverageDailyRevenueMean=("AverageDailyRevenue", "mean"),
        )
        .sort_values(["SystemProfitMean", "HotelRevenueMean"], ascending=False)
    )

    episode_path = FIXED_PRICE_SCAN_DIR / "fixed_price_scan_episode.csv"
    summary_path = FIXED_PRICE_SCAN_DIR / "fixed_price_scan_summary.csv"
    episode_df.to_csv(episode_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=" * 72)
    print("第一阶段：固定价格扫描完成")
    print("=" * 72)
    print(f"Episode-level CSV: {episode_path}")
    print(f"Summary CSV: {summary_path}")
    if not summary_df.empty:
        best = summary_df.iloc[0]
        print(
            "Best by system profit: "
            f"pon={best['FixedOnlinePrice']:.1f}, "
            f"poff={best['FixedOfflinePrice']:.1f}, "
            f"system={best['SystemProfitMean']:.2f}"
        )


if __name__ == "__main__":
    main()
