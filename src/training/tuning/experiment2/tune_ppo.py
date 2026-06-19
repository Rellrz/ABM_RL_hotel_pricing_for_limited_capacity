"""实验二专用：PPO最小调参入口。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from configs.experiment2 import Experiment2Config, TUNING_FIGURES_DIR
from src.training.tuning.experiment2.objective import summarize_trial
from src.training.tuning.experiment2.report import generate_tuning_figures
from src.training.tuning.experiment2.search_space import GLOBAL_BOUNDS, build_refine_bounds, get_tunable_param_names, suggest_ppo_params
from src.training.tuning.experiment2.trial_runner import run_ppo_trial


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="实验二PPO最小调参")
    p.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    p.add_argument("--coarse-trials", type=int, default=24)
    p.add_argument("--refine-trials", type=int, default=12)
    p.add_argument("--coarse-episodes", type=int, default=300)
    p.add_argument("--refine-episodes", type=int, default=600)
    p.add_argument("--final-episodes", type=int, default=1000)
    p.add_argument("--coarse-seeds", type=int, default=1)
    p.add_argument("--refine-seeds", type=int, default=3)
    p.add_argument("--final-seeds", type=int, default=5)
    p.add_argument("--post-eval-episodes", type=int, default=30)
    p.add_argument("--sampler-seed", type=int, default=20260425)
    p.add_argument("--trial-jobs", type=int, default=1, help="coarse/refine 阶段并行trial进程数")
    p.add_argument("--seed-jobs", type=int, default=1, help="每个trial内部并行的seed进程数")
    return p


def load_historical_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "datasets" / "hotel_bookings.csv"
    df = pd.read_csv(path)
    return df[df["hotel"] == "City Hotel"].copy()


def _execute_trial_task(
    stage: str,
    trial_id: int,
    base_config: Experiment2Config,
    historical_data,
    params: Dict[str, float | int | str],
    seeds: List[int],
    train_episodes: int,
    post_eval_episodes: int,
    seed_n_jobs: int,
) -> Dict:
    tr_df, ev_df = run_ppo_trial(
        base_config=base_config,
        historical_data=historical_data,
        trial_id=trial_id,
        params=params,
        seeds=seeds,
        train_episodes=train_episodes,
        post_eval_episodes=post_eval_episodes,
        stage=stage,
        show_seed_progress=False,
        seed_n_jobs=seed_n_jobs,
    )
    row, metrics = summarize_trial(
        trial_id=trial_id,
        stage=stage,
        params=params,
        training_df=tr_df,
        eval_df=ev_df,
    )
    return {"row": row, "metrics": metrics, "tr_df": tr_df, "ev_df": ev_df}


def _run_stage(
    stage: str,
    n_trials: int,
    base_config: Experiment2Config,
    historical_data,
    train_episodes: int,
    post_eval_episodes: int,
    n_seeds: int,
    sampler_seed: int,
    start_trial_id: int,
    prior_df: pd.DataFrame | None = None,
    trial_n_jobs: int = 1,
    seed_n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bounds = GLOBAL_BOUNDS if stage == "coarse" else build_refine_bounds(prior_df if prior_df is not None else pd.DataFrame())
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=sampler_seed))
    trial_rows: List[Dict] = []
    train_parts: List[pd.DataFrame] = []
    eval_parts: List[pd.DataFrame] = []
    seeds = list(range(1, int(n_seeds) + 1))

    best_so_far: float | None = None
    n_trial_jobs = max(1, int(trial_n_jobs))

    if n_trial_jobs <= 1 or int(n_trials) <= 1:
        trial_iter = tqdm(range(int(n_trials)), desc=f"[{stage}] trials", unit="trial")
        for i in trial_iter:
            trial = study.ask()
            params = suggest_ppo_params(trial, bounds=bounds)
            trial_id = int(start_trial_id + i)
            tqdm.write(f"[TUNE][{stage}] trial={trial_id} params={params}")
            tr_df, ev_df = run_ppo_trial(
                base_config=base_config,
                historical_data=historical_data,
                trial_id=trial_id,
                params=params,
                seeds=seeds,
                train_episodes=train_episodes,
                post_eval_episodes=post_eval_episodes,
                stage=stage,
                show_seed_progress=True,
                seed_n_jobs=seed_n_jobs,
            )
            row, metrics = summarize_trial(
                trial_id=trial_id,
                stage=stage,
                params=params,
                training_df=tr_df,
                eval_df=ev_df,
            )
            row["OptunaTrialNumber"] = int(trial.number)
            trial_rows.append(row)
            train_parts.append(tr_df)
            eval_parts.append(ev_df)
            score = float(metrics["Score"])
            study.tell(trial, score)
            best_so_far = score if best_so_far is None else max(best_so_far, score)
            trial_iter.set_postfix(
                trial=trial_id,
                score=f"{score:.3f}",
                best=f"{best_so_far:.3f}",
                stable=int(bool(row.get("Stable", False))),
            )
    else:
        max_workers = min(n_trial_jobs, int(n_trials))
        trial_bar = tqdm(total=int(n_trials), desc=f"[{stage}] trials", unit="trial")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            in_flight: Dict = {}
            next_idx = 0
            while next_idx < int(n_trials) and len(in_flight) < max_workers:
                trial = study.ask()
                params = suggest_ppo_params(trial, bounds=bounds)
                trial_id = int(start_trial_id + next_idx)
                tqdm.write(f"[TUNE][{stage}] trial={trial_id} params={params}")
                fut = ex.submit(
                    _execute_trial_task,
                    stage,
                    trial_id,
                    base_config,
                    historical_data,
                    params,
                    seeds,
                    train_episodes,
                    post_eval_episodes,
                    int(seed_n_jobs),
                )
                in_flight[fut] = (trial, trial_id)
                next_idx += 1

            while in_flight:
                done_future = next(as_completed(in_flight))
                trial, trial_id = in_flight.pop(done_future)
                payload = done_future.result()
                row = payload["row"]
                metrics = payload["metrics"]
                tr_df = payload["tr_df"]
                ev_df = payload["ev_df"]
                row["OptunaTrialNumber"] = int(trial.number)
                trial_rows.append(row)
                train_parts.append(tr_df)
                eval_parts.append(ev_df)
                score = float(metrics["Score"])
                study.tell(trial, score)
                best_so_far = score if best_so_far is None else max(best_so_far, score)
                trial_bar.update(1)
                trial_bar.set_postfix(
                    trial=trial_id,
                    score=f"{score:.3f}",
                    best=f"{best_so_far:.3f}",
                    stable=int(bool(row.get("Stable", False))),
                    running=len(in_flight),
                )

                if next_idx < int(n_trials):
                    trial_new = study.ask()
                    params_new = suggest_ppo_params(trial_new, bounds=bounds)
                    trial_id_new = int(start_trial_id + next_idx)
                    tqdm.write(f"[TUNE][{stage}] trial={trial_id_new} params={params_new}")
                    fut_new = ex.submit(
                        _execute_trial_task,
                        stage,
                        trial_id_new,
                        base_config,
                        historical_data,
                        params_new,
                        seeds,
                        train_episodes,
                        post_eval_episodes,
                        int(seed_n_jobs),
                    )
                    in_flight[fut_new] = (trial_new, trial_id_new)
                    next_idx += 1
        trial_bar.close()

    return (
        pd.DataFrame(trial_rows),
        pd.concat(train_parts, axis=0, ignore_index=True) if train_parts else pd.DataFrame(),
        pd.concat(eval_parts, axis=0, ignore_index=True) if eval_parts else pd.DataFrame(),
    )


def _best_row(trials_df: pd.DataFrame) -> Dict:
    if trials_df is None or len(trials_df) == 0:
        raise RuntimeError("No trial rows found.")
    stable = trials_df[trials_df["Stable"] == True]  # noqa: E712
    use = stable if len(stable) > 0 else trials_df
    return use.sort_values("Score", ascending=False).iloc[0].to_dict()


def main() -> None:
    if importlib.util.find_spec("stable_baselines3") is None:  # pragma: no cover
        raise RuntimeError("未检测到 stable-baselines3，请先安装。")
    args = build_parser().parse_args()
    config = Experiment2Config(run_mode=args.mode)
    config.ensure_dirs()
    historical_data = load_historical_data()

    # 安全保护：避免嵌套ProcessPool导致macOS上挂起
    trial_jobs = max(1, int(args.trial_jobs))
    seed_jobs = max(1, int(args.seed_jobs))
    if trial_jobs > 1 and seed_jobs > 1:
        print(
            f"[WARN] 检测到嵌套并行 trial_jobs={trial_jobs} + seed_jobs={seed_jobs}。"
            f"为防止macOS multiprocessing挂起，自动将 seed_jobs 降为 1。"
        )
        seed_jobs = 1
        args.seed_jobs = 1

    print("=" * 72)
    print("实验二：PPO最小调参")
    print("=" * 72)
    print(
        f"mode={args.mode} coarse={args.coarse_trials} refine={args.refine_trials} "
        f"trial_jobs={max(1, int(args.trial_jobs))} seed_jobs={max(1, int(args.seed_jobs))}"
    )

    coarse_trials_df, coarse_train_df, coarse_eval_df = _run_stage(
        stage="coarse",
        n_trials=args.coarse_trials,
        base_config=config,
        historical_data=historical_data,
        train_episodes=args.coarse_episodes,
        post_eval_episodes=args.post_eval_episodes,
        n_seeds=args.coarse_seeds,
        sampler_seed=args.sampler_seed,
        start_trial_id=1,
        prior_df=None,
        trial_n_jobs=args.trial_jobs,
        seed_n_jobs=args.seed_jobs,
    )
    refine_trials_df, refine_train_df, refine_eval_df = _run_stage(
        stage="refine",
        n_trials=args.refine_trials,
        base_config=config,
        historical_data=historical_data,
        train_episodes=args.refine_episodes,
        post_eval_episodes=args.post_eval_episodes,
        n_seeds=args.refine_seeds,
        sampler_seed=args.sampler_seed + 1,
        start_trial_id=1 + args.coarse_trials,
        prior_df=coarse_trials_df,
        trial_n_jobs=args.trial_jobs,
        seed_n_jobs=args.seed_jobs,
    )

    trials_df = pd.concat([coarse_trials_df, refine_trials_df], axis=0, ignore_index=True)
    train_df = pd.concat([coarse_train_df, refine_train_df], axis=0, ignore_index=True)
    eval_df = pd.concat([coarse_eval_df, refine_eval_df], axis=0, ignore_index=True)
    best = _best_row(trials_df)
    best_trial_id = int(best["TrialID"])
    best_params: Dict[str, float | int | str] = {}
    for col in get_tunable_param_names():
        if col in best:
            val = best[col]
            if isinstance(val, float) and col not in ("ppo_reward_mode",):
                best_params[col] = float(val)
            elif col in ("ppo_n_steps", "ppo_batch_size"):
                best_params[col] = int(val)
            else:
                best_params[col] = val

    final_seeds = list(range(1, int(args.final_seeds) + 1))
    baseline_params: Dict[str, float | int | str] = {}
    for col in get_tunable_param_names():
        default_val = getattr(config, col, None)
        if default_val is not None:
            baseline_params[col] = default_val
    final_bar = tqdm(total=2, desc="[final] validation", unit="run")
    final_baseline_train, final_baseline_eval = run_ppo_trial(
        base_config=config,
        historical_data=historical_data,
        trial_id=-1,
        params=baseline_params,
        seeds=final_seeds,
        train_episodes=args.final_episodes,
        post_eval_episodes=args.post_eval_episodes,
        stage="final_baseline",
        show_seed_progress=True,
        seed_n_jobs=args.seed_jobs,
    )
    final_bar.update(1)
    final_bar.set_postfix(last="baseline")
    final_best_train, final_best_eval = run_ppo_trial(
        base_config=config,
        historical_data=historical_data,
        trial_id=-2,
        params=best_params,
        seeds=final_seeds,
        train_episodes=args.final_episodes,
        post_eval_episodes=args.post_eval_episodes,
        stage="final_best",
        show_seed_progress=True,
        seed_n_jobs=args.seed_jobs,
    )
    final_bar.update(1)
    final_bar.set_postfix(last="best")
    final_bar.close()
    final_train_df = pd.concat([final_baseline_train, final_best_train], axis=0, ignore_index=True)
    final_eval_df = pd.concat([final_baseline_eval, final_best_eval], axis=0, ignore_index=True)
    full_train_df = pd.concat([train_df, final_train_df], axis=0, ignore_index=True)
    full_eval_df = pd.concat([eval_df, final_eval_df], axis=0, ignore_index=True)

    trials_df.to_csv(config.tuning_trials_csv_path, index=False)
    full_train_df.to_csv(config.tuning_train_csv_path, index=False)
    full_eval_df.to_csv(config.tuning_eval_csv_path, index=False)

    generate_tuning_figures(
        trials_df=trials_df,
        training_df=full_train_df,
        eval_df=full_eval_df,
        out_dir=TUNING_FIGURES_DIR,
        best_trial_id=-2,
        baseline_trial_id=-1,
    )

    best_json = {
        "best_trial_id": best_trial_id,
        "best_stage": best["Stage"],
        "best_score": float(best["Score"]),
        "best_stable": bool(best["Stable"]),
        "best_params": best_params,
        "coarse_trials": int(args.coarse_trials),
        "refine_trials": int(args.refine_trials),
    }
    with open(config.tuning_best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    summary = {
        "mode": args.mode,
        "n_trials_total": int(len(trials_df)),
        "n_training_rows": int(len(full_train_df)),
        "n_eval_rows": int(len(full_eval_df)),
        "files": {
            "trials_csv": str(config.tuning_trials_csv_path),
            "train_csv": str(config.tuning_train_csv_path),
            "eval_csv": str(config.tuning_eval_csv_path),
            "best_json": str(config.tuning_best_json_path),
            "figures_dir": str(TUNING_FIGURES_DIR),
        },
    }
    with open(config.tuning_summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[TUNE] 完成，汇总: {config.tuning_summary_json_path}")


if __name__ == "__main__":
    main()
