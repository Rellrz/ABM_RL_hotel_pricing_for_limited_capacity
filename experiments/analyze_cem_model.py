from __future__ import annotations

import argparse
import ast
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "analysis"

STATE_COLUMNS = [
    "stage_id",
    "season",
    "weekday",
    "near_inv_bin",
    "far_inv_bin",
]

STATE_VALUE_LABELS = {
    "stage_id": {
        0: "checkin_day",
        1: "1",
        2: "2-3",
        3: "4-6",
        4: "7-13",
        5: "14-29",
        6: "30-59",
        7: "60-90",
    },
    "season": {0: "low", 1: "mid", 2: "high"},
    "weekday": {0: "workday", 1: "weekend"},
    "near_inv_bin": {0: "very_low", 1: "low", 2: "mid", 3: "high", 4: "very_high"},
    "far_inv_bin": {0: "very_low", 1: "low", 2: "mid", 3: "high", 4: "very_high"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析最新生成的CEM联合定价模型。")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="模型JSON路径；不传则自动读取 outputs/models 中最新的 hotel_joint_agent_*.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="分析结果输出目录",
    )
    return parser


def find_latest_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("hotel_joint_agent_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"未找到模型文件: {model_dir}")
    return candidates[0]


def parse_state_key(raw_key: str) -> Tuple[int, ...]:
    parsed = ast.literal_eval(raw_key)
    if isinstance(parsed, tuple):
        return tuple(int(x) for x in parsed)
    raise ValueError(f"当前脚本仅支持元组状态键，收到: {raw_key}")


def load_model_rows(model_path: Path) -> Tuple[Dict, pd.DataFrame]:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    means = payload.get("means", {})
    visits = payload.get("state_visit_count", {})

    rows: List[Dict] = []
    for raw_state_key, prices in means.items():
        state_tuple = parse_state_key(raw_state_key)
        row = {
            "state_key": raw_state_key,
            "visit_count": int(visits.get(raw_state_key, 0)),
            "online_mean_price": float(prices[0]),
            "offline_mean_price": float(prices[1]),
            "offline_minus_online": float(prices[1] - prices[0]),
        }
        for col, value in zip(STATE_COLUMNS, state_tuple):
            row[col] = int(value)
        rows.append(row)

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise ValueError(f"模型中没有可分析的 means: {model_path}")

    detail_df = detail_df.sort_values(["visit_count", "stage_id"], ascending=[False, True]).reset_index(drop=True)
    return payload, detail_df


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    values_series = pd.Series(list(values), dtype=float)
    weights_series = pd.Series(list(weights), dtype=float)
    total_weight = float(weights_series.sum())
    if total_weight <= 0:
        return float(values_series.mean())
    return float((values_series * weights_series).sum() / total_weight)


def build_dimension_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict] = []
    for dim in STATE_COLUMNS:
        dim_df = detail_df[[dim, "visit_count", "online_mean_price", "offline_mean_price"]].copy()
        for state_value, group in dim_df.groupby(dim, sort=True):
            state_value = int(state_value)
            summary_rows.append(
                {
                    "dimension": dim,
                    "state_value": state_value,
                    "state_label": STATE_VALUE_LABELS.get(dim, {}).get(state_value, str(state_value)),
                    "n_states": int(len(group)),
                    "total_visit_count": int(group["visit_count"].sum()),
                    "mean_visit_count": float(group["visit_count"].mean()),
                    "online_mean_price_unweighted": float(group["online_mean_price"].mean()),
                    "offline_mean_price_unweighted": float(group["offline_mean_price"].mean()),
                    "online_mean_price_weighted": weighted_mean(group["online_mean_price"], group["visit_count"]),
                    "offline_mean_price_weighted": weighted_mean(group["offline_mean_price"], group["visit_count"]),
                }
            )
    out_df = pd.DataFrame(summary_rows)
    return out_df.sort_values(["dimension", "state_value"]).reset_index(drop=True)


def build_pair_dimension_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict] = []
    for dim_1, dim_2 in itertools.combinations(STATE_COLUMNS, 2):
        cols = [dim_1, dim_2, "visit_count", "online_mean_price", "offline_mean_price"]
        pair_df = detail_df[cols].copy()
        grouped = pair_df.groupby([dim_1, dim_2], sort=True)
        for (value_1, value_2), group in grouped:
            value_1 = int(value_1)
            value_2 = int(value_2)
            summary_rows.append(
                {
                    "dimension_1": dim_1,
                    "state_value_1": value_1,
                    "state_label_1": STATE_VALUE_LABELS.get(dim_1, {}).get(value_1, str(value_1)),
                    "dimension_2": dim_2,
                    "state_value_2": value_2,
                    "state_label_2": STATE_VALUE_LABELS.get(dim_2, {}).get(value_2, str(value_2)),
                    "n_states": int(len(group)),
                    "total_visit_count": int(group["visit_count"].sum()),
                    "mean_visit_count": float(group["visit_count"].mean()),
                    "online_mean_price_unweighted": float(group["online_mean_price"].mean()),
                    "offline_mean_price_unweighted": float(group["offline_mean_price"].mean()),
                    "online_mean_price_weighted": weighted_mean(group["online_mean_price"], group["visit_count"]),
                    "offline_mean_price_weighted": weighted_mean(group["offline_mean_price"], group["visit_count"]),
                }
            )
    out_df = pd.DataFrame(summary_rows)
    return out_df.sort_values(
        ["dimension_1", "dimension_2", "state_value_1", "state_value_2"]
    ).reset_index(drop=True)


def build_top_states(detail_df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    cols = [
        "state_key",
        "visit_count",
        "online_mean_price",
        "offline_mean_price",
        "offline_minus_online",
        *STATE_COLUMNS,
    ]
    return detail_df[cols].head(top_k).copy()


def save_outputs(
    *,
    model_path: Path,
    output_dir: Path,
    detail_df: pd.DataFrame,
    dimension_df: pd.DataFrame,
    pair_dimension_df: pd.DataFrame,
    top_states_df: pd.DataFrame,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem
    paths = {
        "detail": output_dir / f"{stem}_state_detail.csv",
        "dimension": output_dir / f"{stem}_dimension_summary.csv",
        "pair_dimension": output_dir / f"{stem}_pair_dimension_summary.csv",
        "top_states": output_dir / f"{stem}_top_states.csv",
    }
    detail_df.to_csv(paths["detail"], index=False)
    dimension_df.to_csv(paths["dimension"], index=False)
    pair_dimension_df.to_csv(paths["pair_dimension"], index=False)
    top_states_df.to_csv(paths["top_states"], index=False)
    return paths


def print_console_summary(
    model_path: Path,
    payload: Dict,
    detail_df: pd.DataFrame,
    dimension_df: pd.DataFrame,
    pair_dimension_df: pd.DataFrame,
) -> None:
    print("=" * 72)
    print("CEM模型分析完成")
    print("=" * 72)
    print(f"模型文件: {model_path}")
    print(f"算法: {payload.get('algo', 'unknown')}")
    print(f"已学习状态数: {len(detail_df)}")
    print(f"总访问次数: {int(detail_df['visit_count'].sum())}")
    print(f"online均价: {detail_df['online_mean_price'].mean():.2f}")
    print(f"offline均价: {detail_df['offline_mean_price'].mean():.2f}")
    print("-" * 72)
    for dim in STATE_COLUMNS:
        sub = dimension_df[dimension_df["dimension"] == dim].copy()
        print(f"[{dim}]")
        print(sub[["state_value", "state_label", "total_visit_count", "offline_mean_price_weighted", "online_mean_price_weighted"]].to_string(index=False))
        print("-" * 72)
    focus_pairs = [
        ("stage_id", "far_inv_bin"),
        ("near_inv_bin", "far_inv_bin"),
    ]
    for dim_1, dim_2 in focus_pairs:
        sub = pair_dimension_df[
            (pair_dimension_df["dimension_1"] == dim_1)
            & (pair_dimension_df["dimension_2"] == dim_2)
        ].copy()
        if sub.empty:
            continue
        print(f"[{dim_1} x {dim_2}]")
        print(
            sub[
                [
                    "state_value_1",
                    "state_label_1",
                    "state_value_2",
                    "state_label_2",
                    "total_visit_count",
                    "offline_mean_price_weighted",
                    "online_mean_price_weighted",
                ]
            ].to_string(index=False)
        )
        print("-" * 72)


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model_path.resolve() if args.model_path else find_latest_model(DEFAULT_MODEL_DIR)
    output_dir = args.output_dir.resolve()

    payload, detail_df = load_model_rows(model_path)
    dimension_df = build_dimension_summary(detail_df)
    pair_dimension_df = build_pair_dimension_summary(detail_df)
    top_states_df = build_top_states(detail_df)
    paths = save_outputs(
        model_path=model_path,
        output_dir=output_dir,
        detail_df=detail_df,
        dimension_df=dimension_df,
        pair_dimension_df=pair_dimension_df,
        top_states_df=top_states_df,
    )
    print_console_summary(model_path, payload, detail_df, dimension_df, pair_dimension_df)
    print("输出文件:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
