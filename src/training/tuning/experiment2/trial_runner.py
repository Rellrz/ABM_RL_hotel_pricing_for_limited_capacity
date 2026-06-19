"""执行单个PPO调参trial。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, Iterable, Tuple

import pandas as pd
from tqdm import tqdm

from configs.experiment2 import Experiment2Config
from src.training.ppo_baseline import run_ppo_single_seed


def _run_one_seed(cfg: Experiment2Config, historical_data, seed: int, algorithm_name: str):
    tr, ev, _ = run_ppo_single_seed(
        config=cfg,
        historical_data=historical_data,
        seed=int(seed),
        show_progress=False,
        algorithm_name=algorithm_name,
    )
    return tr, ev, int(seed)


def _attach_trial_meta(tr_records, ev_records, trial_id: int, stage: str) -> None:
    for r in tr_records:
        r["TrialID"] = int(trial_id)
        r["Stage"] = str(stage)
    for r in ev_records:
        r["TrialID"] = int(trial_id)
        r["Stage"] = str(stage)


def run_ppo_trial(
    base_config: Experiment2Config,
    historical_data,
    trial_id: int,
    params: Dict[str, float | int | str],
    seeds: Iterable[int],
    train_episodes: int,
    post_eval_episodes: int,
    stage: str,
    show_seed_progress: bool = True,
    seed_n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = deepcopy(base_config)
    cfg.override_train_episodes = int(train_episodes)
    cfg.post_eval_episodes = int(post_eval_episodes)
    cfg.ppo_use_sde = False

    for k, v in params.items():
        setattr(cfg, k, v)

    all_train = []
    all_eval = []
    algo_name = f"PPO_TRIAL_{int(trial_id)}"
    seed_list = [int(s) for s in seeds]
    n_jobs = max(1, int(seed_n_jobs))

    if n_jobs <= 1 or len(seed_list) <= 1:
        seed_iter = (
            tqdm(
                seed_list,
                desc=f"[{stage}] trial={int(trial_id)} seeds",
                unit="seed",
                leave=False,
                disable=not show_seed_progress,
            )
            if show_seed_progress
            else seed_list
        )
        for sd in seed_iter:
            tr, ev, _ = _run_one_seed(cfg, historical_data, sd, algo_name)
            _attach_trial_meta(tr_records=tr, ev_records=ev, trial_id=trial_id, stage=stage)
            all_train.extend(tr)
            all_eval.extend(ev)
    else:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(seed_list))) as ex:
            future_map = {
                ex.submit(_run_one_seed, cfg, historical_data, sd, algo_name): sd for sd in seed_list
            }
            with tqdm(
                total=len(seed_list),
                desc=f"[{stage}] trial={int(trial_id)} seeds",
                unit="seed",
                leave=False,
                disable=not show_seed_progress,
            ) as pbar:
                done = 0
                for fut in as_completed(future_map):
                    tr, ev, sd = fut.result()
                    _attach_trial_meta(tr_records=tr, ev_records=ev, trial_id=trial_id, stage=stage)
                    all_train.extend(tr)
                    all_eval.extend(ev)
                    done += 1
                    pbar.update(1)
                    pbar.set_postfix(done=f"{done}/{len(seed_list)}", last=sd)

    return pd.DataFrame(all_train), pd.DataFrame(all_eval)
