# Repository Guidelines

## Project Structure & Module Organization

This repository currently implements the baseline model described in `idea2.md`. The current `idea2` baseline uses a two-segment customer ABM: `biz` customers mostly stick to their ideal stay date, while `flex` customers can shift across the three-day window in response to price. Treat `idea.md` as a future extension, not as part of the current acceptance scope.

- `configs/config.py`: dataclass-based data, ABM, environment, PPO, SAC, and path settings.
- `src/environment/`: customer ABM, core hotel simulation, and Gymnasium wrapper.
- `src/utils/preprocess_data.py`: historical-data loading and demand calibration.
- `src/training/train_ppo.py`: reusable Stable-Baselines3 PPO trainer implementation.
- `src/training/train_sac.py`: reusable Stable-Baselines3 SAC trainer implementation with bounded continuous actions handled by the algorithm.
- `src/training/algorithm_registry.py`: central algorithm registry that maps `ppo` / `sac` to their training and evaluation builders.
- `experiments/experiment_train_ppo.py`: simple single-run training entry point; now supports `--algo`.
- `experiments/experiment_train_sac.py`: simple single-run training entry point; now supports `--algo`.
- `experiments/experiment_capacity_sensitivity.py`: capacity sensitivity experiment entry point with training, evaluation, CSV export, and single-metric plots. It supports multi-process execution across capacities and `--algo`.
- `experiments/experiment_penalty_scaling.py`: penalty scaling experiment entry point; supports `--algo`.
- `experiments/experiment_penalty_ablation.py`: penalty ablation experiment entry point; supports `--algo`.
- `experiments/experiment_ppo_benchmark.py`: learned-policy benchmark entry point for hard upper bound plus strong baselines. Despite the legacy filename, it now supports both `--algo` and `--algos` for multi-algorithm comparison in one run.
- `train_ppo.py`: thin compatibility wrapper that forwards to `src/training/train_ppo.py`.
- `datasets/`: source dataset and exploratory notebook.
- `outputs/`: generated models, experiment summaries, and TensorBoard logs; avoid committing new run artifacts unless required for reproducibility.

Add automated tests under `tests/`, mirroring source modules, for example `tests/environment/test_hotel_env.py`.

## Build, Test, and Development Commands

Create an isolated Python environment before installing dependencies.

On this machine, run project commands through the existing Conda environment:

```bash
conda run -n abm_new python ...
```

```bash
pip install -r requirements.txt
python experiments/experiment_train_ppo.py --algo ppo
python experiments/experiment_train_sac.py --algo sac
python experiments/experiment_capacity_sensitivity.py --algo ppo --capacities 20 30 40 50 60
python experiments/experiment_capacity_sensitivity.py --algo sac --capacities 20 30 40 50 60
python experiments/experiment_ppo_benchmark.py --algos ppo sac --max-workers 5
tensorboard --logdir outputs/tensorboard
python -m compileall -q configs src experiments train_ppo.py
```

The training commands use the active values in `configs/config.py`. For quick validation, temporarily reduce `total_timesteps` and `episode_days`, but do not commit debug-only values. When an experiment script supports both `--algo` and `--algos`, prefer `--algos` only for benchmarks or direct cross-algorithm comparison; use `--algo` for simpler single-algorithm sweeps.

When tests are added, run them with:

```bash
python -m pytest -q
```

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, type hints, and `from __future__ import annotations`. Follow PEP 8 conventions:

- modules, functions, and variables: `snake_case`
- classes and dataclasses: `PascalCase`
- constants and shared configuration objects: `UPPER_SNAKE_CASE`

Keep simulation logic in `src/environment`, calibration logic in `src/utils`, reusable trainer logic in `src/training`, and experiment entry points in `experiments`. Prefer NumPy generators over global random state, and preserve explicit seed handling.

Additional collaboration and implementation rules:

- Do not write compatibility code unless the user explicitly asks for backward compatibility or multi-version support.
- Before writing code, first describe the implementation plan and wait for explicit user approval.
- When requirements are ambiguous, ask clarifying questions before writing code.
- If a change will touch more than 3 files, first split the work into smaller tasks and align on that breakdown.
- After each user correction, reflect on the cause and state a concrete plan to avoid repeating the same mistake.

When running capacity experiments, prefer `ProcessPoolExecutor`-style multi-process parallelism over threads. These runs are CPU-heavy because each worker launches a full RL training loop and environment simulation.

## Testing Guidelines

Use `pytest`. Name files `test_*.py` and tests `test_<behavior>()`. Prioritize deterministic tests for inventory rolling, capacity truncation, reference-price updates, reward calculation, action scaling, seeded resets, and experiment result aggregation. Any behavioral change should include a regression test and a short fixed-seed smoke run.

## Commit & Pull Request Guidelines

History uses short, imperative summaries, often in Chinese, such as `完成最小实现`. Keep each commit focused and avoid committing caches, `.DS_Store`, or routine training outputs.

Pull requests should state the research assumption being changed, list affected modules and configuration values, include validation commands, and summarize fixed-seed results. Attach plots only when training behavior or experimental conclusions change.

## Research and Configuration Discipline

Separate empirical calibration from scenario assumptions. Document demand multipliers, capacity stress settings, exclusions, train/eval random seeds, and experiment-specific overrides in saved run configuration. Do not silently reinterpret long-lead bookings as arrivals inside the three-day `idea2` window.

For routine research iteration, prefer changing `configs/config.py`, experiment script arguments, or explicit experiment-level overrides before modifying files under `src/`. Only change `src/` when the current code path cannot express the intended experiment through parameters alone, or when fixing a confirmed modeling/implementation bug. When `src/` changes are necessary, keep them minimal, explain why parameter-only control was insufficient, and avoid mixing reusable logic changes with one-off experimental tuning.

For algorithm experimentation, prefer adding new trainers behind `src/training/algorithm_registry.py` and reusing the shared `train_single_run` / `build_eval_env` interface rather than hard-coding `if algo == ...` branches throughout experiment scripts. Keep experiment outputs algorithm-labeled in run names, CSV rows, and summary files.

For the benchmark script, note that multi-process runs with `--max-workers > 1` may not show Stable-Baselines3's per-worker progress bars cleanly in the main terminal. In the current multi-algorithm benchmark, each worker computes baseline searches before launching RL training, so visible training progress may appear delayed even when training is running correctly.

For the current `idea2` customer segmentation, treat `flexible_customer_share`, `lambda_day_mismatch_biz`, and `lambda_day_mismatch_flex` as scenario parameters first, not as directly identified facts from the booking data. When studying cross-day substitution, prefer sweeping these configuration values in experiments before adding more ABM complexity.

Current research findings indicate two important cautions:

- Reward penalties alone do not explain the extreme price structures observed in experiments; cross-day substitution in the ABM is currently the dominant driver of "low-price funnel + high-price blocking" behavior.
- `day0` being relatively cheaper is not automatically a bug, because it has the shortest remaining selling horizon. The modeling concern is when this mild dynamic-pricing effect is amplified into extreme cross-day funneling.
- In the segmented-demand version of `idea2`, the key structural question is how much substitution comes from the `flex` segment versus how much rigid demand is anchored by the `biz` segment. Keep that distinction explicit in experiment design and result interpretation.

If you change the reward design, keep the configuration explicit in `configs/config.py` and preserve separate logging for revenue, total penalty, full-capacity penalty, and scarcity penalty so experimental comparisons remain interpretable.
