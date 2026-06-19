#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用参数敏感性实验，支持单参数和多参数组合。

示例：
单参数
conda run -n abm python experiments/run_single_param_sensitivity.py \
  --param RL_CONFIG.cem_alpha \
  --values 0.1,0.2,0.3 \
  --episodes 100 \
  --tail-window 30 \
  --n-jobs 3

多参数-笛卡尔积
conda run -n abm python experiments/run_single_param_sensitivity.py \
  --param-values RL_CONFIG.cem_alpha=0.1,0.2,0.3 \
  --param-values ABM_CONFIG.anchor_eta=0.1,0.2 \
  --episodes 100 \
  --tail-window 30 \
  --n-jobs 4

多参数-自定义组合（每个参数组合需要有引号引起来）
conda run -n abm python experiments/run_single_param_sensitivity.py \
  --combo 'RL_CONFIG.reward_shape_price_weight=0.0;RL_CONFIG.reward_shape_sellthrough_weight=0.0;RL_CONFIG.reward_shape_target_sellthrough=0' \
  --combo 'RL_CONFIG.reward_shape_price_weight=0.3;RL_CONFIG.reward_shape_sellthrough_weight=0.22;RL_CONFIG.reward_shape_target_sellthrough=0.25' \
  --episodes 100 \
  --tail-window 30 \
  --n-jobs 2
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import ABM_CONFIG, ENV_CONFIG, PATH_CONFIG, RANDOM_CONFIG, RL_CONFIG


CONFIG_REGISTRY = {
    "RL_CONFIG": RL_CONFIG,
    "ABM_CONFIG": ABM_CONFIG,
    "ENV_CONFIG": ENV_CONFIG,
    "RANDOM_CONFIG": RANDOM_CONFIG,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用参数敏感性实验（支持单参数和多参数组合）")
    parser.add_argument("--data", type=str, default="datasets/hotel_bookings.csv", help="酒店预订数据文件路径")
    parser.add_argument("--param", type=str, default=None, help="单参数模式的目标参数，如 RL_CONFIG.cem_alpha")
    parser.add_argument("--values", type=str, default=None, help="单参数模式的候选值，如 0.1,0.2,0.3")
    parser.add_argument(
        "--param-values",
        action="append",
        default=[],
        help="参数取值定义，可重复传入，如 RL_CONFIG.cem_alpha=0.1,0.2,0.3",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="显式参数组合，可重复传入，如 'RL_CONFIG.cem_alpha=0.1;ABM_CONFIG.anchor_eta=0.2'",
    )
    parser.add_argument("--episodes", type=int, default=100, help="训练轮数")
    parser.add_argument("--episode-days", type=int, default=730, help="每个 episode 的模拟天数")
    parser.add_argument("--mode", type=str, default="simultaneous", choices=["fixed_ota", "alternating", "simultaneous"], help="训练模式")
    parser.add_argument("--update-frequency", type=int, default=30, help="CEM 参数更新频率")
    parser.add_argument("--booking-window-days", type=int, default=91, help="预订窗口长度")
    parser.add_argument("--decision-buckets", type=str, default="0|1|2-3|4-6|7-13|14-29|30-59|60-90", help="提前期分桶")
    parser.add_argument("--tail-window", type=int, default=30, help="收敛段统计窗口长度")
    parser.add_argument("--env-seed", type=int, default=42, help="固定环境随机种子")
    parser.add_argument("--n-jobs", type=int, default=4, help="并行进程数")
    parser.add_argument("--save-models", action="store_true", help="是否保存每个参数值对应的模型")
    return parser


def _tail_mean(series: pd.Series, window: int) -> float:
    n = max(1, min(len(series), int(window)))
    return float(series.tail(n).mean())


def _tail_std(series: pd.Series, window: int) -> float:
    n = max(1, min(len(series), int(window)))
    return float(series.tail(n).std(ddof=0))


def _parse_target(param_expr: str) -> tuple[str, str]:
    if "." not in str(param_expr):
        raise ValueError(f"参数写法错误，应为 CONFIG_NAME.field，收到: {param_expr}")
    config_name, field_name = str(param_expr).split(".", 1)
    config_name = config_name.strip()
    field_name = field_name.strip()
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"不支持的配置对象: {config_name}，可选: {sorted(CONFIG_REGISTRY)}")
    if not hasattr(CONFIG_REGISTRY[config_name], field_name):
        raise ValueError(f"{config_name} 中不存在字段: {field_name}")
    return config_name, field_name


def _coerce_single_value(raw_value: str, template_value: Any) -> Any:
    raw = str(raw_value).strip()
    if isinstance(template_value, bool):
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"无法解析布尔值: {raw_value}")
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(raw)
    if isinstance(template_value, float):
        return float(raw)
    if template_value is None:
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


def _parse_values(values_expr: str, template_value: Any) -> list[Any]:
    values = [token.strip() for token in str(values_expr).split(",") if token.strip()]
    if not values:
        raise ValueError("values 不能为空")
    return [_coerce_single_value(token, template_value) for token in values]


def _parse_param_values_specs(
    single_param: str | None,
    single_values: str | None,
    param_values_specs: list[str],
) -> list[tuple[str, str, list[Any]]]:
    parsed_specs: list[tuple[str, str, list[Any]]] = []

    if single_param is not None or single_values is not None:
        if not single_param or not single_values:
            raise ValueError("使用 --param / --values 时，两个参数都必须提供")
        config_name, field_name = _parse_target(single_param)
        template_value = getattr(CONFIG_REGISTRY[config_name], field_name)
        parsed_specs.append((config_name, field_name, _parse_values(single_values, template_value)))

    for raw_spec in param_values_specs:
        if "=" not in str(raw_spec):
            raise ValueError(f"--param-values 格式错误，应为 CONFIG.field=v1,v2，收到: {raw_spec}")
        param_expr, values_expr = str(raw_spec).split("=", 1)
        config_name, field_name = _parse_target(param_expr.strip())
        template_value = getattr(CONFIG_REGISTRY[config_name], field_name)
        parsed_specs.append((config_name, field_name, _parse_values(values_expr.strip(), template_value)))

    if not parsed_specs:
        raise ValueError("请提供 --param/--values 或至少一个 --param-values")

    deduped: list[tuple[str, str, list[Any]]] = []
    seen_targets: set[tuple[str, str]] = set()
    for config_name, field_name, values in parsed_specs:
        key = (config_name, field_name)
        if key in seen_targets:
            raise ValueError(f"重复定义参数: {config_name}.{field_name}")
        seen_targets.add(key)
        deduped.append((config_name, field_name, values))
    return deduped


def _parse_single_assignment(raw_assignment: str) -> tuple[str, Any]:
    if "=" not in str(raw_assignment):
        raise ValueError(f"组合中的参数写法错误，应为 CONFIG.field=value，收到: {raw_assignment}")
    param_expr, raw_value = str(raw_assignment).split("=", 1)
    config_name, field_name = _parse_target(param_expr.strip())
    template_value = getattr(CONFIG_REGISTRY[config_name], field_name)
    value = _coerce_single_value(raw_value.strip(), template_value)
    return f"{config_name}.{field_name}", value


def _parse_explicit_combos(combo_specs: list[str]) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for raw_combo in combo_specs:
        parts = [token.strip() for token in str(raw_combo).split(";") if token.strip()]
        if not parts:
            raise ValueError(f"空组合定义: {raw_combo}")
        assignments: dict[str, Any] = {}
        for part in parts:
            key, value = _parse_single_assignment(part)
            if key in assignments:
                raise ValueError(f"同一组合内重复定义参数: {key}")
            assignments[key] = value
        combos.append(assignments)
    return combos


def _build_experiment_combinations(
    param_specs: list[tuple[str, str, list[Any]]],
) -> list[dict[str, Any]]:
    # 单参数模式：每个候选值对应一个实验
    if len(param_specs) == 1:
        config_name, field_name, values = param_specs[0]
        return [{f"{config_name}.{field_name}": value} for value in values]

    raise ValueError(
        "多参数时请使用 --combo 显式给定组合；当前脚本不再默认对多个参数做笛卡尔积。"
    )


def _format_combo_label(assignments: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in assignments.items()]
    return " | ".join(parts)


def _safe_combo_stub(assignments: dict[str, Any]) -> str:
    parts = []
    for key, value in assignments.items():
        safe_key = key.replace(".", "_")
        safe_value = str(value).replace(".", "p").replace("/", "_").replace("|", "_")
        parts.append(f"{safe_key}_{safe_value}")
    return "__".join(parts)


def _run_single_value(
    assignments: dict[str, Any],
    args_dict: dict,
) -> tuple[dict, pd.DataFrame]:
    from src.training.game_trainer import train_game_system

    historical_data = pd.read_csv(args_dict["data"])
    historical_data = historical_data[historical_data["hotel"] == "City Hotel"].copy()

    old_random_mode = RANDOM_CONFIG.random_mode
    old_fixed_seed = RANDOM_CONFIG.fixed_seed
    old_values: dict[tuple[str, str], Any] = {}
    combo_label = _format_combo_label(assignments)

    try:
        RANDOM_CONFIG.random_mode = "fixed"
        RANDOM_CONFIG.fixed_seed = int(args_dict["env_seed"])
        np.random.seed(int(args_dict["env_seed"]))
        for param_key, candidate_value in assignments.items():
            config_name, field_name = _parse_target(param_key)
            target_config = CONFIG_REGISTRY[config_name]
            old_values[(config_name, field_name)] = getattr(target_config, field_name)
            setattr(target_config, field_name, candidate_value)

        hotel_agent, _, _, _, episode_info = train_game_system(
            historical_data=historical_data,
            episodes=int(args_dict["episodes"]),
            training_mode=str(args_dict["mode"]),
            update_frequency=int(args_dict["update_frequency"]),
            booking_window_days=int(args_dict["booking_window_days"]),
            decision_buckets=str(args_dict["decision_buckets"]),
            episode_days=int(args_dict["episode_days"]),
        )

        model_path = ""
        if bool(args_dict.get("save_models", False)) and getattr(hotel_agent, "cem_joint", None) is not None:
            model_path = str(hotel_agent.cem_joint.save_model(f"hotel_joint_{_safe_combo_stub(assignments)}"))

        df = pd.DataFrame(episode_info)
        df["param_name"] = combo_label
        df["param_value"] = combo_label
        df["param_combo"] = combo_label
        for param_key, candidate_value in assignments.items():
            df[param_key.replace(".", "__")] = str(candidate_value)

        summary_row = {
            "param_name": combo_label,
            "param_value": combo_label,
            "param_combo": combo_label,
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
            "subsidy_tail_mean": _tail_mean(df["total_subsidy"], int(args_dict["tail_window"])),
            "model_path": model_path,
        }
        for param_key, candidate_value in assignments.items():
            summary_row[param_key.replace(".", "__")] = str(candidate_value)
        return summary_row, df
    finally:
        for (config_name, field_name), old_value in old_values.items():
            setattr(CONFIG_REGISTRY[config_name], field_name, old_value)
        RANDOM_CONFIG.random_mode = old_random_mode
        RANDOM_CONFIG.fixed_seed = old_fixed_seed


def main() -> None:
    args = build_parser().parse_args()
    explicit_combos = _parse_explicit_combos(list(args.combo))
    param_specs: list[tuple[str, str, list[Any]]] = []
    if explicit_combos:
        experiment_combinations = explicit_combos
    else:
        param_specs = _parse_param_values_specs(args.param, args.values, list(args.param_values))
        experiment_combinations = _build_experiment_combinations(param_specs)

    summary_rows: list[dict] = []
    episode_frames: list[pd.DataFrame] = []
    max_workers = max(1, min(int(args.n_jobs), len(experiment_combinations)))
    args_dict = vars(args).copy()

    print("=" * 72)
    print("参数敏感性实验")
    print("=" * 72)
    if explicit_combos:
        print("显式参数组合:")
        for idx, assignments in enumerate(experiment_combinations, start=1):
            print(f"  - combo_{idx}: {_format_combo_label(assignments)}")
    else:
        print("参数定义:")
        for config_name, field_name, values in param_specs:
            print(f"  - {config_name}.{field_name}: {values}")
    print(f"组合数量: {len(experiment_combinations)}")
    print(f"环境种子: {args.env_seed}")
    print(f"训练轮数: {args.episodes}")
    print(f"并行进程数: {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_single_value, assignments, args_dict): assignments
            for assignments in experiment_combinations
        }
        for fut in as_completed(futures):
            assignments = futures[fut]
            print("\n" + "-" * 72)
            print(f"完成参数组合: {_format_combo_label(assignments)}")
            print("-" * 72)
            summary_row, df = fut.result()
            summary_rows.append(summary_row)
            episode_frames.append(df)

    summary_df = pd.DataFrame(summary_rows).sort_values("hotel_tail_mean", ascending=False).reset_index(drop=True)
    episode_df = pd.concat(episode_frames, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if explicit_combos:
        param_stub = "explicit_combos"
    else:
        param_stub = "__".join(f"{config_name}_{field_name}" for config_name, field_name, _ in param_specs)
    summary_path = os.path.join(PATH_CONFIG.results_dir, f"single_param_sensitivity_summary_{param_stub}_{timestamp}.csv")
    detail_path = os.path.join(PATH_CONFIG.results_dir, f"single_param_sensitivity_episodes_{param_stub}_{timestamp}.csv")
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
