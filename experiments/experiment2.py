"""实验二主入口：对比与消融实验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment2 import Experiment2Config
from src.evaluation.experiment_stats import build_performance_table, significance_tests
from src.plot.experiment2_plotter import plot_learning_curves, plot_post_eval_bar
from src.training.bo_baseline import run_bo
from src.training.cem_multivariate_baseline import run_multivariate_cem
from src.training.cem_independent_baseline import run_independent_cem
from src.training.emsrb_baseline import run_emsrb
from src.training.ga_baseline import run_ga
from src.training.oracle_baseline import run_oracle
from src.training.rs_baseline import run_rs
from src.training.sa_baseline import run_sa
from src.training.ppo_baseline import run_ppo
from src.training.qlearning_baseline import run_qlearning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实验二：对比与消融")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-qlearning", action="store_true")
    parser.add_argument("--skip-cem", action="store_true", help="同时跳过 Multivariate CEM 和 Independent CEM")
    parser.add_argument("--skip-cem-mv", action="store_true", help="跳过 Multivariate CEM")
    parser.add_argument("--skip-cem-ind", action="store_true", help="跳过 Independent CEM")
    parser.add_argument("--skip-emsrb", action="store_true")
    parser.add_argument("--skip-bo", action="store_true")
    parser.add_argument("--skip-ga", action="store_true")
    parser.add_argument("--skip-sa", action="store_true")
    parser.add_argument("--skip-rs", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--ppo-reward-mode", type=str, default=None, choices=["scaled_raw_daily", "raw_daily", "shaped_bucket", "mixed"])
    parser.add_argument("--ppo-reward-scale", type=float, default=None)
    parser.add_argument("--ppo-shaped-reward-weight", type=float, default=None)
    return parser


def load_historical_data(project_root: Path) -> pd.DataFrame:
    path = project_root / "datasets" / "hotel_bookings.csv"
    df = pd.read_csv(path)
    return df[df["hotel"] == "City Hotel"].copy()


def run_experiment2(
    config: Experiment2Config,
    historical_data: pd.DataFrame,
    skip_ppo: bool = False,
    skip_qlearning: bool = False,
    skip_cem: bool = False,
    skip_cem_mv: bool = False,
    skip_cem_ind: bool = False,
    skip_emsrb: bool = False,
    skip_bo: bool = False,
    skip_ga: bool = False,
    skip_sa: bool = False,
    skip_rs: bool = False,
    skip_oracle: bool = False,
) -> dict:
    config.ensure_dirs()
    training_records = []
    eval_records = []

    if not skip_emsrb:
        rec_train, rec_eval = run_emsrb(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    if not skip_oracle:
        rec_train, rec_eval = run_oracle(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    if not skip_cem and not skip_cem_mv:
        rec_train, rec_eval = run_multivariate_cem(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    
    if not skip_cem and not skip_cem_ind:
        rec_train, rec_eval = run_independent_cem(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    
    if not skip_bo:
        rec_train, rec_eval = run_bo(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    if not skip_ga:
        rec_train, rec_eval = run_ga(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    if not skip_sa:
        rec_train, rec_eval = run_sa(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    if not skip_rs:
        rec_train, rec_eval = run_rs(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    if not skip_qlearning:
        rec_train, rec_eval = run_qlearning(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    if not skip_ppo:
        rec_train, rec_eval = run_ppo(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)

    train_columns = [
        "Algorithm", "Seed", "Episode", "EpisodeHotelRevenue", "EpisodeOTAProfit", "EpisodeSystemProfit", "EpisodeRevenue"
    ]
    eval_columns = [
        "Algorithm", "Seed", "EvalEpisode", "EvalHotelRevenue", "EvalOTAProfit", "EvalSystemProfit", "EvalRevenue"
    ]
    train_df = pd.DataFrame(training_records) if training_records else pd.DataFrame(columns=train_columns)
    eval_df = pd.DataFrame(eval_records) if eval_records else pd.DataFrame(columns=eval_columns)
    train_df.to_csv(config.training_csv_path, index=False)
    eval_df.to_csv(config.evaluation_csv_path, index=False)

    perf_hotel_df = build_performance_table(train_df, eval_df, "EpisodeHotelRevenue", "EvalHotelRevenue", "Hotel Revenue")
    perf_hotel_df.to_csv(config.performance_table_hotel_csv, index=False)
    perf_ota_df = build_performance_table(train_df, eval_df, "EpisodeOTAProfit", "EvalOTAProfit", "OTA Profit")
    perf_ota_df.to_csv(config.performance_table_ota_csv, index=False)
    perf_system_df = build_performance_table(train_df, eval_df, "EpisodeSystemProfit", "EvalSystemProfit", "System Profit")
    perf_system_df.to_csv(config.performance_table_system_csv, index=False)

    stats_hotel_df = significance_tests(eval_df, eval_metric_col="EvalHotelRevenue")
    stats_hotel_df.to_csv(config.stats_hotel_csv_path, index=False)
    stats_ota_df = significance_tests(eval_df, eval_metric_col="EvalOTAProfit")
    stats_ota_df.to_csv(config.stats_ota_csv_path, index=False)
    stats_system_df = significance_tests(eval_df, eval_metric_col="EvalSystemProfit")
    stats_system_df.to_csv(config.stats_system_csv_path, index=False)

    plot_learning_curves(config, train_df, "EpisodeHotelRevenue", "Episode Hotel Revenue", config.learning_curve_hotel_pdf)
    plot_learning_curves(config, train_df, "EpisodeOTAProfit", "Episode OTA Profit", config.learning_curve_ota_pdf)
    plot_learning_curves(config, train_df, "EpisodeSystemProfit", "Episode System Profit", config.learning_curve_system_pdf)
    plot_post_eval_bar(config, eval_df, "EvalHotelRevenue", "Post-Training Evaluation Hotel Revenue", config.eval_bar_hotel_pdf)
    plot_post_eval_bar(config, eval_df, "EvalOTAProfit", "Post-Training Evaluation OTA Profit", config.eval_bar_ota_pdf)
    plot_post_eval_bar(config, eval_df, "EvalSystemProfit", "Post-Training Evaluation System Profit", config.eval_bar_system_pdf)

    summary = {
        "mode": config.run_mode,
        "n_training_records": int(len(train_df)),
        "n_eval_records": int(len(eval_df)),
        "algorithms": sorted(set(train_df.get("Algorithm", []).tolist()) | set(eval_df.get("Algorithm", []).tolist())),
        "training_csv": str(config.training_csv_path),
        "evaluation_csv": str(config.evaluation_csv_path),
        "outputs_dir": str(config.training_csv_path.parent.parent),
    }
    with open(config.summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    config = Experiment2Config(run_mode=args.mode)
    if args.n_jobs is not None:
        config.n_jobs = int(args.n_jobs)
    if args.ppo_reward_mode is not None:
        config.ppo_reward_mode = str(args.ppo_reward_mode)
    if args.ppo_reward_scale is not None:
        config.ppo_reward_scale = float(args.ppo_reward_scale)
    if args.ppo_shaped_reward_weight is not None:
        config.ppo_shaped_reward_weight = float(args.ppo_shaped_reward_weight)
    historical_data = load_historical_data(PROJECT_ROOT)
    summary = run_experiment2(
        config=config,
        historical_data=historical_data,
        skip_ppo=bool(args.skip_ppo),
        skip_qlearning=bool(args.skip_qlearning),
        skip_cem=bool(args.skip_cem),
        skip_cem_mv=bool(args.skip_cem_mv),
        skip_cem_ind=bool(args.skip_cem_ind),
        skip_emsrb=bool(args.skip_emsrb),
        skip_bo=bool(args.skip_bo),
        skip_ga=bool(args.skip_ga),
        skip_sa=bool(args.skip_sa),
        skip_rs=bool(args.skip_rs),
        skip_oracle=bool(args.skip_oracle),
    )
    print(f"实验二完成，结果输出目录: {summary['outputs_dir']}")


if __name__ == "__main__":
    main()
