from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase1_utils import (
    CONFIG_SNAPSHOT_DIR,
    DIAGNOSTICS_DIR,
    build_runtime_snapshot,
    ensure_phase1_dirs,
    load_city_hotel_data,
    save_json,
)
from configs.config import ABM_CONFIG
from configs.experiment2 import Experiment2Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第一阶段：环境诊断与配置快照导出")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    return parser


def summarize_city_hotel_dataset(df: pd.DataFrame) -> dict:
    stays = (
        df["stays_in_week_nights"].fillna(0).astype(float)
        + df["stays_in_weekend_nights"].fillna(0).astype(float)
    )
    out = {
        "n_rows": int(len(df)),
        "cancel_rate": float(df["is_canceled"].mean()) if "is_canceled" in df.columns else None,
        "adr_mean": float(df["adr"].mean()) if "adr" in df.columns else None,
        "adr_std": float(df["adr"].std()) if "adr" in df.columns else None,
        "lead_time_mean": float(df["lead_time"].mean()) if "lead_time" in df.columns else None,
        "lead_time_std": float(df["lead_time"].std()) if "lead_time" in df.columns else None,
        "stay_nights_mean": float(stays.mean()),
        "stay_nights_std": float(stays.std()),
    }
    return out


def build_arrival_rate_table() -> pd.DataFrame:
    rows = []
    for month in sorted((ABM_CONFIG.arrival_rate_by_month_daytype or {}).keys()):
        by_type = ABM_CONFIG.arrival_rate_by_month_daytype.get(month, {})
        rows.append(
            {
                "month": int(month),
                "workday_arrival_rate": float(by_type.get(0, ABM_CONFIG.monthly_arrival_rates.get(month, 100.0))),
                "holiday_arrival_rate": float(by_type.get(1, ABM_CONFIG.monthly_arrival_rates.get(month, 100.0))),
                "monthly_arrival_rate": float(ABM_CONFIG.monthly_arrival_rates.get(month, 100.0)),
            }
        )
    return pd.DataFrame(rows)


def build_lead_time_summary() -> dict:
    params = ABM_CONFIG.lead_time_params or {}
    support = params.get("support", [])
    empirical = params.get("probabilities", [])
    conditional = params.get("conditional_probabilities", {})
    n_conditional_groups = 0
    if isinstance(conditional, dict):
        for season_map in conditional.values():
            if isinstance(season_map, dict):
                n_conditional_groups += len(season_map)
    return {
        "type": params.get("type"),
        "mean": params.get("mean"),
        "max_lead_time": params.get("max_days"),
        "n_support_points": int(len(support)),
        "n_empirical_points": int(len(empirical)),
        "n_conditional_groups": int(n_conditional_groups),
    }


def build_wtp_summary() -> dict:
    params = ABM_CONFIG.wtp_params or {}
    by_group = params.get("by_season_weekday", {})
    return {
        "mean": params.get("mean"),
        "std": params.get("std"),
        "season_weekday_groups": int(len(by_group)),
    }


def main() -> None:
    args = build_parser().parse_args()
    ensure_phase1_dirs()

    config = Experiment2Config(run_mode=args.mode)
    df = load_city_hotel_data()

    dataset_summary = summarize_city_hotel_dataset(df)
    lead_time_summary = build_lead_time_summary()
    wtp_summary = build_wtp_summary()
    arrival_rate_table = build_arrival_rate_table()
    runtime_snapshot = build_runtime_snapshot(config)

    save_json(dataset_summary, DIAGNOSTICS_DIR / "city_hotel_dataset_summary.json")
    save_json(lead_time_summary, DIAGNOSTICS_DIR / "lead_time_summary.json")
    save_json(wtp_summary, DIAGNOSTICS_DIR / "wtp_summary.json")
    save_json(runtime_snapshot, CONFIG_SNAPSHOT_DIR / "runtime_config_snapshot.json")
    arrival_rate_table.to_csv(DIAGNOSTICS_DIR / "arrival_rate_month_daytype.csv", index=False)

    print("=" * 72)
    print("第一阶段：环境诊断完成")
    print("=" * 72)
    print(f"输出目录: {Path(DIAGNOSTICS_DIR)}")
    print(f"配置快照: {CONFIG_SNAPSHOT_DIR / 'runtime_config_snapshot.json'}")


if __name__ == "__main__":
    main()
