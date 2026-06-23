# Repository Guidelines

## Project Structure & Module Organization

This repository currently implements the baseline model described in `idea2.md`. Treat `idea.md` as a future extension, not as part of the current acceptance scope.

- `configs/config.py`: dataclass-based data, ABM, environment, PPO, and path settings.
- `src/environment/`: customer ABM, core hotel simulation, and Gymnasium wrapper.
- `src/utils/preprocess_data.py`: historical-data loading and demand calibration.
- `train_ppo.py`: Stable-Baselines3 PPO training entry point.
- `datasets/`: source dataset and exploratory notebook.
- `outputs/`: generated models and TensorBoard logs; avoid committing new run artifacts unless required for reproducibility.

Add automated tests under `tests/`, mirroring source modules, for example `tests/environment/test_hotel_env.py`.

## Build, Test, and Development Commands

Create an isolated Python environment before installing dependencies.

```bash
pip install -r requirements.txt
python train_ppo.py
tensorboard --logdir outputs/tensorboard
python -m compileall -q configs src train_ppo.py
```

The training command uses the active values in `configs/config.py`. For quick validation, temporarily reduce `total_timesteps` and `episode_days`, but do not commit debug-only values.

When tests are added, run them with:

```bash
python -m pytest -q
```

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, type hints, and `from __future__ import annotations`. Follow PEP 8 conventions:

- modules, functions, and variables: `snake_case`
- classes and dataclasses: `PascalCase`
- constants and shared configuration objects: `UPPER_SNAKE_CASE`

Keep simulation logic in `src/environment`, calibration logic in `src/utils`, and experiment parameters in `configs/config.py`. Prefer NumPy generators over global random state, and preserve explicit seed handling.

## Testing Guidelines

Use `pytest`. Name files `test_*.py` and tests `test_<behavior>()`. Prioritize deterministic tests for inventory rolling, capacity truncation, reference-price updates, reward calculation, action scaling, and seeded resets. Any behavioral change should include a regression test and a short fixed-seed smoke run.

## Commit & Pull Request Guidelines

History uses short, imperative summaries, often in Chinese, such as `完成最小实现`. Keep each commit focused and avoid committing caches, `.DS_Store`, or routine training outputs.

Pull requests should state the research assumption being changed, list affected modules and configuration values, include validation commands, and summarize fixed-seed results. Attach plots only when training behavior or experimental conclusions change.

## Research and Configuration Discipline

Separate empirical calibration from scenario assumptions. Document demand multipliers, capacity stress settings, exclusions, and random seeds in saved run configuration. Do not silently reinterpret long-lead bookings as arrivals inside the three-day `idea2` window.
