# Repository Guidelines

## Project Structure & Module Organization

This repository currently implements the baseline model described in `idea2.md`. Treat `idea.md` as a future extension, not as part of the current acceptance scope.

- `configs/config.py`: dataclass-based data, ABM, environment, PPO, and path settings.
- `src/environment/`: customer ABM, core hotel simulation, and Gymnasium wrapper.
- `src/utils/preprocess_data.py`: historical-data loading and demand calibration.
- `src/training/train_ppo.py`: reusable Stable-Baselines3 PPO trainer implementation.
- `experiments/experiment_train_ppo.py`: simple single-run PPO experiment entry point.
- `experiments/experiment_capacity_sensitivity.py`: capacity sensitivity experiment entry point with training, evaluation, CSV export, and single-metric plots. It supports multi-process execution across capacities.
- `train_ppo.py`: thin compatibility wrapper that forwards to `src/training/train_ppo.py`.
- `datasets/`: source dataset and exploratory notebook.
- `outputs/`: generated models, experiment summaries, and TensorBoard logs; avoid committing new run artifacts unless required for reproducibility.

Add automated tests under `tests/`, mirroring source modules, for example `tests/environment/test_hotel_env.py`.

## Build, Test, and Development Commands

Create an isolated Python environment before installing dependencies.

```bash
pip install -r requirements.txt
python experiments/experiment_train_ppo.py
python experiments/experiment_capacity_sensitivity.py --capacities 20 30 40 50 60
python experiments/experiment_capacity_sensitivity.py --capacities 20 30 40 50 60 --max-workers 2 --no-progress-bar
tensorboard --logdir outputs/tensorboard
python -m compileall -q configs src experiments train_ppo.py
```

The training commands use the active values in `configs/config.py`. For quick validation, temporarily reduce `total_timesteps` and `episode_days`, but do not commit debug-only values.

When tests are added, run them with:

```bash
python -m pytest -q
```

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, type hints, and `from __future__ import annotations`. Follow PEP 8 conventions:

- modules, functions, and variables: `snake_case`
- classes and dataclasses: `PascalCase`
- constants and shared configuration objects: `UPPER_SNAKE_CASE`

Keep simulation logic in `src/environment`, calibration logic in `src/utils`, reusable PPO training logic in `src/training`, and experiment entry points in `experiments`. Prefer NumPy generators over global random state, and preserve explicit seed handling.

When running capacity experiments, prefer `ProcessPoolExecutor`-style multi-process parallelism over threads. These runs are CPU-heavy because each worker launches a full PPO training loop and environment simulation.

## Testing Guidelines

Use `pytest`. Name files `test_*.py` and tests `test_<behavior>()`. Prioritize deterministic tests for inventory rolling, capacity truncation, reference-price updates, reward calculation, action scaling, seeded resets, and experiment result aggregation. Any behavioral change should include a regression test and a short fixed-seed smoke run.

## Commit & Pull Request Guidelines

History uses short, imperative summaries, often in Chinese, such as `完成最小实现`. Keep each commit focused and avoid committing caches, `.DS_Store`, or routine training outputs.

Pull requests should state the research assumption being changed, list affected modules and configuration values, include validation commands, and summarize fixed-seed results. Attach plots only when training behavior or experimental conclusions change.

## Research and Configuration Discipline

Separate empirical calibration from scenario assumptions. Document demand multipliers, capacity stress settings, exclusions, train/eval random seeds, and experiment-specific overrides in saved run configuration. Do not silently reinterpret long-lead bookings as arrivals inside the three-day `idea2` window.

Current research findings indicate two important cautions:

- Reward penalties alone do not explain the extreme price structures observed in experiments; cross-day substitution in the ABM is currently the dominant driver of "low-price funnel + high-price blocking" behavior.
- `day0` being relatively cheaper is not automatically a bug, because it has the shortest remaining selling horizon. The modeling concern is when this mild dynamic-pricing effect is amplified into extreme cross-day funneling.

If you change the reward design, keep the configuration explicit in `configs/config.py` and preserve separate logging for revenue, total penalty, full-capacity penalty, and scarcity penalty so experimental comparisons remain interpretable.
