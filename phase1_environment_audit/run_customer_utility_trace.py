from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_utils import CUSTOMER_UTILITY_TRACE_DIR, ensure_phase1_dirs, load_city_hotel_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment2 import Experiment2Config
from src.environment.bucket_pricing_simulator import BucketPricingSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第一阶段：单episode消费者效用轨迹导出")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--online-price", type=float, default=100.0)
    parser.add_argument("--offline-price", type=float, default=120.0)
    return parser


def run_single_episode_trace(
    config: Experiment2Config,
    historical_data,
    seed: int,
    online_price: float,
    offline_price: float,
) -> tuple[pd.DataFrame, dict]:
    sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
    sim.env.abm_model.trace_customer_utility = True
    sim.reset()

    total_hotel = 0.0
    total_ota = 0.0
    done = False
    while not done:
        stage_actions = [(online_price, offline_price)] * sim.n_stages
        out = sim.step_day(stage_actions)
        total_hotel += float(out.reward_hotel)
        total_ota += float(out.reward_ota)
        done = bool(out.done)

    trace_df = sim.env.abm_model.get_customer_utility_trace()
    summary = {
        "HotelRevenue": float(total_hotel),
        "OTAProfit": float(total_ota),
        "SystemProfit": float(total_hotel + total_ota),
        "NumRows": int(len(trace_df)),
        "NumBooked": int(trace_df["booked"].sum()) if len(trace_df) > 0 else 0,
    }
    return trace_df, summary


def build_lead_time_summary(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df is None or len(trace_df) == 0:
        return pd.DataFrame(
            columns=[
                "lead_time",
                "n_customers",
                "online_utility_mean",
                "offline_utility_mean",
                "chosen_utility_mean",
                "utility_threshold_pass_rate",
                "booking_success_rate",
                "online_choice_rate",
                "offline_choice_rate",
            ]
        )

    work = trace_df.copy()
    work["online_choice"] = (work["chosen_channel"] == "online").astype(float)
    work["offline_choice"] = (work["chosen_channel"] == "offline").astype(float)

    summary_df = (
        work.groupby("lead_time", as_index=False)
        .agg(
            n_customers=("customer_id", "count"),
            online_utility_mean=("online_utility", "mean"),
            offline_utility_mean=("offline_utility", "mean"),
            chosen_utility_mean=("chosen_utility", "mean"),
            utility_threshold_pass_rate=("passed_utility_threshold", "mean"),
            booking_success_rate=("booked", "mean"),
            online_choice_rate=("online_choice", "mean"),
            offline_choice_rate=("offline_choice", "mean"),
        )
        .sort_values("lead_time")
    )
    return summary_df


def main() -> None:
    args = build_parser().parse_args()
    ensure_phase1_dirs()

    config = Experiment2Config(run_mode=args.mode)
    historical_data = load_city_hotel_data()
    online_price = float(np.clip(args.online_price, config.online_price_min, config.online_price_max))
    offline_price = float(np.clip(args.offline_price, config.offline_price_min, config.offline_price_max))

    trace_df, summary = run_single_episode_trace(
        config=config,
        historical_data=historical_data,
        seed=int(args.seed),
        online_price=online_price,
        offline_price=offline_price,
    )
    lead_time_df = build_lead_time_summary(trace_df)

    stem = f"seed{int(args.seed)}_pon{online_price:.1f}_poff{offline_price:.1f}".replace(".", "p")
    trace_path = CUSTOMER_UTILITY_TRACE_DIR / f"customer_utility_trace_episode_{stem}.csv"
    lead_time_path = CUSTOMER_UTILITY_TRACE_DIR / f"customer_utility_trace_by_lead_time_{stem}.csv"
    trace_df.to_csv(trace_path, index=False)
    lead_time_df.to_csv(lead_time_path, index=False)

    print("=" * 72)
    print("第一阶段：单episode消费者效用轨迹导出完成")
    print("=" * 72)
    print(f"Episode trace CSV: {trace_path}")
    print(f"Lead-time summary CSV: {lead_time_path}")
    print(
        f"Rows={summary['NumRows']}, Booked={summary['NumBooked']}, "
        f"Hotel={summary['HotelRevenue']:.2f}, OTA={summary['OTAProfit']:.2f}"
    )


if __name__ == "__main__":
    main()
