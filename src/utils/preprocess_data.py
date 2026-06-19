from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from configs.config import ABM_CONFIG, DATA_CONFIG, PATH_CONFIG


@dataclass
class DemandCalibration:
    weekday_arrival_mean: float
    weekend_arrival_mean: float
    ideal_offset_probs: np.ndarray
    weekday_ref_price: float
    weekend_ref_price: float
    weekday_wtp_mean: float
    weekday_wtp_std: float
    weekend_wtp_mean: float
    weekend_wtp_std: float


def load_filtered_historical_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    path = PATH_CONFIG.data_path if csv_path is None else Path(csv_path)
    usecols = [
        "hotel",
        "lead_time",
        DATA_CONFIG.adr_column,
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["hotel"] == DATA_CONFIG.hotel_name].copy()
    df = df[df[DATA_CONFIG.adr_column].fillna(0.0) > 0.0].copy()
    return df.reset_index(drop=True)


def build_demand_calibration(historical_data: pd.DataFrame) -> DemandCalibration:
    if historical_data.empty:
        return DemandCalibration(
            weekday_arrival_mean=float(ABM_CONFIG.weekday_arrival_fallback_mean),
            weekend_arrival_mean=float(ABM_CONFIG.weekend_arrival_fallback_mean),
            ideal_offset_probs=np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=float),
            weekday_ref_price=120.0,
            weekend_ref_price=150.0,
            weekday_wtp_mean=140.0,
            weekday_wtp_std=20.0,
            weekend_wtp_mean=170.0,
            weekend_wtp_std=25.0,
        )

    df = historical_data.copy()
    arrival_date = pd.to_datetime(
        dict(
            year=df["arrival_date_year"].astype(int),
            month=pd.to_datetime(df["arrival_date_month"], format="%B").dt.month.astype(int),
            day=df["arrival_date_day_of_month"].astype(int),
        ),
        errors="coerce",
    )
    df = df.loc[arrival_date.notna()].copy()
    df["arrival_date"] = arrival_date.loc[arrival_date.notna()]
    df["lead_time"] = df["lead_time"].fillna(0).astype(int).clip(lower=0)
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["stay_is_weekend"] = (df["arrival_date"].dt.weekday >= 5).astype(int)

    booking_counts = df.groupby("booking_date").size().rename("n_arrivals")
    if booking_counts.empty:
        weekday_arrival_mean = float(ABM_CONFIG.weekday_arrival_fallback_mean)
        weekend_arrival_mean = float(ABM_CONFIG.weekend_arrival_fallback_mean)
    else:
        full_booking_dates = pd.date_range(
            start=booking_counts.index.min(),
            end=booking_counts.index.max(),
            freq="D",
        )
        booking_counts = booking_counts.reindex(full_booking_dates, fill_value=0).reset_index()
        booking_counts.columns = ["booking_date", "n_arrivals"]
        booking_counts["booking_is_weekend"] = (booking_counts["booking_date"].dt.weekday >= 5).astype(int)

        weekday_mean_series = booking_counts.loc[booking_counts["booking_is_weekend"] == 0, "n_arrivals"]
        weekend_mean_series = booking_counts.loc[booking_counts["booking_is_weekend"] == 1, "n_arrivals"]
        weekday_arrival_mean = float(
            weekday_mean_series.mean() if not weekday_mean_series.empty else ABM_CONFIG.weekday_arrival_fallback_mean
        )
        weekend_arrival_mean = float(
            weekend_mean_series.mean() if not weekend_mean_series.empty else ABM_CONFIG.weekend_arrival_fallback_mean
        )

    lead_counts = (
        df["lead_time"].value_counts().reindex([0, 1, 2], fill_value=0.0).to_numpy(dtype=float)
    )
    if float(lead_counts.sum()) <= 0.0:
        lead_counts = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        lead_counts = lead_counts / lead_counts.sum()

    ref_prices = df.groupby("stay_is_weekend")[DATA_CONFIG.adr_column].mean()
    wtp_stats = df.groupby("stay_is_weekend")[DATA_CONFIG.adr_column].agg(["mean", "std"])

    overall_mean = float(df[DATA_CONFIG.adr_column].mean())
    weekday_ref = float(ref_prices.get(0, overall_mean))
    weekend_ref = float(ref_prices.get(1, overall_mean))
    weekday_mean = float(wtp_stats["mean"].get(0, weekday_ref))
    weekday_std = float(wtp_stats["std"].get(0, max(10.0, 0.1 * weekday_mean)))
    weekend_mean = float(wtp_stats["mean"].get(1, weekend_ref))
    weekend_std = float(wtp_stats["std"].get(1, max(10.0, 0.1 * weekend_mean)))

    return DemandCalibration(
        weekday_arrival_mean=max(0.0, weekday_arrival_mean),
        weekend_arrival_mean=max(0.0, weekend_arrival_mean),
        ideal_offset_probs=lead_counts.astype(float),
        weekday_ref_price=max(ABM_CONFIG.wtp_min, weekday_ref),
        weekend_ref_price=max(ABM_CONFIG.wtp_min, weekend_ref),
        weekday_wtp_mean=max(ABM_CONFIG.wtp_min, weekday_mean),
        weekday_wtp_std=max(1.0, weekday_std),
        weekend_wtp_mean=max(ABM_CONFIG.wtp_min, weekend_mean),
        weekend_wtp_std=max(1.0, weekend_std),
    )


__all__ = [
    "DemandCalibration",
    "load_filtered_historical_data",
    "build_demand_calibration",
]
