#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List

import pandas as pd

from .schema import ABMConfig


def calculate_monthly_arrival_rates(historical_data: pd.DataFrame) -> Dict[int, float]:
    df = historical_data.copy()
    if 'arrival_date_month' not in df.columns:
        return {m: 100.0 for m in range(1, 13)}

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    if {'arrival_date_year', 'arrival_date_day_of_month', 'lead_time'}.issubset(df.columns):
        tmp = df.copy()
        tmp['month_num'] = tmp['arrival_date_month'].map(month_map)
        tmp = tmp[tmp['month_num'].notna()].copy()
        tmp['arrival_date'] = pd.to_datetime(
            dict(
                year=tmp['arrival_date_year'].astype(int),
                month=tmp['month_num'].astype(int),
                day=tmp['arrival_date_day_of_month'].astype(int),
            ),
            errors='coerce',
        )
        tmp = tmp[tmp['arrival_date'].notna()].copy()
        tmp['lead_time_clean'] = tmp['lead_time'].fillna(0).astype(int).clip(lower=0)
        tmp['booking_date'] = tmp['arrival_date'] - pd.to_timedelta(tmp['lead_time_clean'], unit='D')
        tmp = tmp[tmp['booking_date'].notna()].copy()
        tmp['booking_month'] = tmp['booking_date'].dt.month
        tmp['booking_day'] = tmp['booking_date'].dt.date

        daily_counts = tmp.groupby(['booking_month', 'booking_day']).size().reset_index(name='cnt')
        monthly_rates_series = daily_counts.groupby('booking_month')['cnt'].mean()
        monthly_counts = monthly_rates_series.to_dict()
    else:
        df['month_num'] = df['arrival_date_month'].map(month_map)
        monthly_counts = (df.groupby('month_num').size() / 30.0).to_dict()

    monthly_rates = {}
    for month in range(1, 13):
        monthly_rates[month] = float(monthly_counts.get(month, 100.0))
    return monthly_rates


def calculate_arrival_rates_by_month_daytype(historical_data: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """
    按月份 + 日类型估计日均到达率：
    - daytype=0: 工作日
    - daytype=1: 节假日（当前使用周末代理）
    """
    base_monthly = calculate_monthly_arrival_rates(historical_data)
    result: Dict[int, Dict[int, float]] = {m: {0: float(base_monthly[m]), 1: float(base_monthly[m])} for m in range(1, 13)}

    df = historical_data.copy()
    required_cols = {'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'lead_time'}
    if not required_cols.issubset(df.columns):
        return result

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['month_num'] = df['arrival_date_month'].map(month_map)
    df = df[df['month_num'].notna()].copy()
    df['arrival_date'] = pd.to_datetime(
        dict(
            year=df['arrival_date_year'].astype(int),
            month=df['month_num'].astype(int),
            day=df['arrival_date_day_of_month'].astype(int),
        ),
        errors='coerce',
    )
    df = df[df['arrival_date'].notna()].copy()
    df['lead_time_clean'] = df['lead_time'].fillna(0).astype(int).clip(lower=0)
    df['booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time_clean'], unit='D')
    df = df[df['booking_date'].notna()].copy()
    df['booking_month'] = df['booking_date'].dt.month.astype(int)
    df['is_holiday'] = (df['booking_date'].dt.dayofweek >= 5).astype(int)
    df['booking_day'] = df['booking_date'].dt.date

    daily = (
        df.groupby(['booking_month', 'is_holiday', 'booking_day'])
        .size()
        .reset_index(name='cnt')
    )
    rates = daily.groupby(['booking_month', 'is_holiday'])['cnt'].mean()

    for month in range(1, 13):
        for daytype in (0, 1):
            key = (month, daytype)
            if key in rates.index:
                result[month][daytype] = float(rates.loc[key])
    return result


def fit_lead_time_distribution(historical_data: pd.DataFrame) -> Dict[str, float]:
    if 'lead_time' not in historical_data.columns:
        return {'mean': 104.0}
    lead_times = historical_data['lead_time'].dropna()
    lead_times = lead_times[lead_times >= 0]
    if len(lead_times) == 0:
        return {'mean': 104.0}
    return {'mean': float(lead_times.mean())}


def build_empirical_lead_time_distribution(
    historical_data: pd.DataFrame,
    max_lead_time_days: int = 90,
) -> Dict[str, Any]:
    lead_times = historical_data['lead_time'].dropna().astype(int)
    lead_times = lead_times[(lead_times >= 0) & (lead_times <= max_lead_time_days)]

    counts = lead_times.value_counts().sort_index()
    support = list(range(max_lead_time_days + 1))
    total = float(counts.sum())
    probabilities = [float(counts.get(d, 0)) / total for d in support] if total > 0 else [0.0 for _ in support]

    prob_sum = float(sum(probabilities))
    if prob_sum > 0:
        probabilities = [p / prob_sum for p in probabilities]

    result: Dict[str, Any] = {
        'type': 'empirical',
        'max_days': max_lead_time_days,
        'support': support,
        'probabilities': probabilities,
    }

    required_cols = {'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'lead_time'}
    if required_cols.issubset(historical_data.columns):
        df = historical_data.copy()
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        df['month_num'] = df['arrival_date_month'].map(month_map)
        df = df[df['month_num'].notna()].copy()
        df['arrival_date'] = pd.to_datetime(
            dict(
                year=df['arrival_date_year'].astype(int),
                month=df['month_num'].astype(int),
                day=df['arrival_date_day_of_month'].astype(int),
            ),
            errors='coerce',
        )
        df = df[df['arrival_date'].notna()].copy()
        # 只保留预订窗口内的历史样本，避免把 > max_lead_time_days 的样本错误堆叠到窗口上界。
        df['lead_time_clean'] = df['lead_time'].fillna(0).astype(int)
        df = df[(df['lead_time_clean'] >= 0) & (df['lead_time_clean'] <= max_lead_time_days)].copy()
        df['booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time_clean'], unit='D')
        df = df[df['booking_date'].notna()].copy()
        df['booking_month'] = df['booking_date'].dt.month
        df['booking_weekend'] = (df['booking_date'].dt.dayofweek >= 5).astype(int)
        df['season'] = df['booking_month'].map(
            lambda m: 0 if m in [11, 12, 1, 2] else (2 if m in [6, 7, 8] else 1)
        ).astype(int)

        conditional_probabilities: Dict[int, Dict[int, List[float]]] = {0: {}, 1: {}, 2: {}}
        alpha = 1.0
        for season in [0, 1, 2]:
            for weekend in [0, 1]:
                seg = df[(df['season'] == season) & (df['booking_weekend'] == weekend)]
                seg_counts = seg['lead_time_clean'].value_counts().to_dict()
                denom = float(len(seg) + alpha * (max_lead_time_days + 1))
                probs = [((float(seg_counts.get(d, 0)) + alpha) / denom) for d in support]
                conditional_probabilities[season][weekend] = probs
        result['conditional_probabilities'] = conditional_probabilities

    return result


def _default_wtp_params() -> Dict[str, Any]:
    by_sw = {0: {}, 1: {}, 2: {}}
    for season in [0, 1, 2]:
        for is_weekend in [0, 1]:
            by_sw[season][is_weekend] = {'mean': 100.0, 'std': 30.0}
    return {
        'type': 'normal',
        'mean': 100.0,
        'std': 30.0,
        'overall': {'mean': 100.0, 'std': 30.0},
        'by_season_weekday': by_sw,
    }


def fit_wtp_distribution(historical_data: pd.DataFrame) -> Dict[str, Any]:
    df = historical_data.copy()
    if 'is_canceled' in df.columns:
        df = df[df['is_canceled'] == 0]

    adr_values = df['adr'].dropna()
    adr_values = adr_values[(adr_values > 0) & (adr_values < 500)]
    overall_mean = float(adr_values.mean()) if len(adr_values) > 0 else 100.0
    std_raw = float(adr_values.std()) if len(adr_values) > 0 else 30.0
    overall_std = std_raw if std_raw > 0 else 30.0

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    if {'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'}.issubset(df.columns):
        df = df.copy()
        df['month_num'] = df['arrival_date_month'].map(month_map)
        df['date'] = pd.to_datetime(
            dict(
                year=df['arrival_date_year'],
                month=df['month_num'],
                day=df['arrival_date_day_of_month'],
            ),
            errors='coerce',
        )
        df = df[df['date'].notna()].copy()
        df = df[df['adr'].notna()].copy()
        df = df[(df['adr'] > 0) & (df['adr'] < 500)].copy()

        if len(df) > 0:
            df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            df['season'] = df['month_num'].map(lambda m: 0 if m in [11, 12, 1, 2] else (2 if m in [6, 7, 8] else 1)).astype(int)
            by_season_weekday: Dict[int, Dict[int, Dict[str, float]]] = {0: {}, 1: {}, 2: {}}
            grouped = df.groupby(['season', 'is_weekend'])['adr']
            for (season, is_weekend), series in grouped:
                series = series.dropna()
                series = series[(series > 0) & (series < 500)]
                if len(series) == 0:
                    continue
                m = float(series.mean())
                s_raw = float(series.std())
                s = s_raw if s_raw > 0 else overall_std
                by_season_weekday[int(season)][int(is_weekend)] = {'mean': m, 'std': s}

            for season in [0, 1, 2]:
                for is_weekend in [0, 1]:
                    if is_weekend not in by_season_weekday[season]:
                        by_season_weekday[season][is_weekend] = {'mean': overall_mean, 'std': overall_std}

            return {
                'type': 'normal',
                'mean': overall_mean,
                'std': overall_std,
                'overall': {'mean': overall_mean, 'std': overall_std},
                'by_season_weekday': by_season_weekday,
            }

    fallback = _default_wtp_params()
    fallback['mean'] = overall_mean
    fallback['std'] = overall_std
    fallback['overall'] = {'mean': overall_mean, 'std': overall_std}
    for season in [0, 1, 2]:
        for is_weekend in [0, 1]:
            fallback['by_season_weekday'][season][is_weekend] = {'mean': overall_mean, 'std': overall_std}
    return fallback


def create_abm_config(data_path: str = None) -> ABMConfig:
    if data_path is not None:
        historical_data = pd.read_csv(data_path)
        historical_data = historical_data[historical_data['hotel'] == 'City Hotel'].copy()
        monthly_arrival_rates = calculate_monthly_arrival_rates(historical_data)
        arrival_rate_by_month_daytype = calculate_arrival_rates_by_month_daytype(historical_data)
        lead_time_params = build_empirical_lead_time_distribution(historical_data, max_lead_time_days=90)
        lead_time_params['mean'] = fit_lead_time_distribution(historical_data)['mean']
        wtp_params = fit_wtp_distribution(historical_data)
    else:
        monthly_arrival_rates = {m: 100.0 for m in range(1, 13)}
        arrival_rate_by_month_daytype = {m: {0: 100.0, 1: 100.0} for m in range(1, 13)}
        lead_time_params = {'type': 'exponential', 'mean': 104.0}
        wtp_params = {'mean': 100.0, 'std': 30.0}

    return ABMConfig(
        monthly_arrival_rates=monthly_arrival_rates,
        arrival_rate_by_month_daytype=arrival_rate_by_month_daytype,
        lead_time_params=lead_time_params,
        wtp_params=wtp_params,
    )
