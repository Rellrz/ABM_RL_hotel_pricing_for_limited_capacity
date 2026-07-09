# Repository Guidelines

## Project Structure & Module Organization

This repository currently implements the baseline model described in `idea2.md`. The current `idea2` baseline uses a two-segment customer ABM: `biz` customers mostly stick to their ideal stay date, while `flex` customers can shift across the three-day window in response to price. Treat `idea.md` as a future extension, not as part of the current acceptance scope.

- `configs/config.py`: dataclass-based data, train/eval year split, ABM, environment, PPO, SAC, and path settings.
- `configs/scenario_policy_training_scenarios.json`: shared scenario list for scenario-policy training and scarcity-penalty ablation experiments.
- `src/environment/`: customer ABM, core hotel simulation, and Gymnasium wrapper.
- `src/utils/preprocess_data.py`: historical-data loading and demand calibration.
- `src/training/train_ppo.py`: reusable Stable-Baselines3 PPO trainer implementation.
- `src/training/beta_policy.py`: PPO actor-critic policy using a Beta action distribution mapped to bounded continuous pricing actions.
- `src/training/truncated_gaussian_policy.py`: PPO actor-critic policies using truncated Gaussian and scale-adjusted truncated Gaussian action distributions for bounded continuous pricing actions.
- `src/training/train_sac.py`: reusable Stable-Baselines3 SAC trainer implementation with bounded continuous actions handled by the algorithm.
- `src/training/algorithm_registry.py`: central algorithm registry that exposes `ppo_standard`, `ppo_tanh_gaussian`, `ppo_truncated_gaussian`, `ppo_scale_adjusted_truncated_gaussian`, `ppo_beta`, and `sac` as independent experiment algorithms.
- `src/baseline/pricing_baselines.py`: reusable static, weekday/weekend static, and inventory-protection baseline policies and search helpers.
- `experiments/experiment_train_single_algo.py`: simple single-run training entry point; supports `--algo`.
- `experiments/experiment_capacity_sensitivity.py`: capacity sensitivity experiment entry point with training, evaluation, CSV export, and single-metric plots. It supports multi-process execution across capacities and `--algo`.
- `experiments/experiment_penalty_ablation.py`: scarcity penalty coefficient ablation entry point; supports `--algo` and modes such as `scarcity_0`, `scarcity_3000`, `scarcity_6000`, and `scarcity_9000`.
- `experiments/experiment_policy_benchmark.py`: learned-policy benchmark entry point for hard upper bound plus strong baselines; supports both `--algo` and `--algos` for multi-algorithm comparison in one run.
- `experiments/experiment_dynamic_baseline_diagnostics.py`: baseline-only diagnostic entry point for testing whether global static, weekday/weekend static, or inventory-protection policies reveal dynamic pricing room.
- `experiments/experiment_mechanism_diagnostics.py`: mechanism-grid diagnostic entry point that sweeps `flexible_customer_share`, `lambda_day_mismatch_flex`, and capacity without training RL policies.
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
conda run -n abm_new python experiments/experiment_train_single_algo.py --algo ppo_tanh_gaussian
conda run -n abm_new python experiments/experiment_train_single_algo.py --algo sac
conda run -n abm_new python experiments/experiment_capacity_sensitivity.py --algo ppo_tanh_gaussian --capacities 20 30 40 50 60
conda run -n abm_new python experiments/experiment_capacity_sensitivity.py --algo sac --capacities 20 30 40 50 60
conda run -n abm_new python experiments/experiment_penalty_ablation.py --algo ppo_beta --modes scarcity_0 scarcity_3000 scarcity_6000 scarcity_9000 --eval-seeds 142 143 144
conda run -n abm_new python experiments/experiment_scenario_policy_training.py --scenario-file configs/scenario_policy_training_scenarios.json --algos sac ppo_beta
conda run -n abm_new python experiments/experiment_policy_benchmark.py --algos ppo_tanh_gaussian sac --max-workers 5
conda run -n abm_new python experiments/experiment_dynamic_baseline_diagnostics.py --max-workers 5
conda run -n abm_new python experiments/experiment_mechanism_diagnostics.py --max-workers 6
tensorboard --logdir outputs/tensorboard
conda run -n abm_new python -m compileall -q configs src experiments
```

The training commands use the active values in `configs/config.py`. For quick validation, temporarily reduce `total_timesteps` and `episode_days`, but do not commit debug-only values. When an experiment script supports both `--algo` and `--algos`, prefer `--algos` only for benchmarks or direct cross-algorithm comparison; use `--algo` for simpler single-algorithm sweeps.

When tests are added, run them with:

```bash
conda run -n abm_new python -m pytest -q
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
- Keep this `AGENTS.md` file current. When project structure, canonical commands, experiment entry points, environment assumptions, or collaboration rules change, update `AGENTS.md` in the same task so future work starts from the correct shared context.

When running capacity experiments, prefer `ProcessPoolExecutor`-style multi-process parallelism over threads. These runs are CPU-heavy because each worker launches a full RL training loop and environment simulation.

## Testing Guidelines

Use `pytest`. Name files `test_*.py` and tests `test_<behavior>()`. Prioritize deterministic tests for inventory rolling, capacity truncation, reference-price updates, reward calculation, action scaling, seeded resets, and experiment result aggregation. Any behavioral change should include a regression test and a short fixed-seed smoke run.

## Commit & Pull Request Guidelines

History uses short, imperative summaries, often in Chinese, such as `完成最小实现`. Keep each commit focused and avoid committing caches, `.DS_Store`, or routine training outputs.

Pull requests should state the research assumption being changed, list affected modules and configuration values, include validation commands, and summarize fixed-seed results. Attach plots only when training behavior or experimental conclusions change.

## Research and Configuration Discipline

Separate empirical calibration from scenario assumptions. Document demand multipliers, capacity stress settings, exclusions, train/eval random seeds, and experiment-specific overrides in saved run configuration. Do not silently reinterpret long-lead bookings as arrivals inside the three-day `idea2` window.

For routine research iteration, prefer changing `configs/config.py`, experiment script arguments, or explicit experiment-level overrides before modifying files under `src/`. Only change `src/` when the current code path cannot express the intended experiment through parameters alone, or when fixing a confirmed modeling/implementation bug. When `src/` changes are necessary, keep them minimal, explain why parameter-only control was insufficient, and avoid mixing reusable logic changes with one-off experimental tuning.

The canonical temporal generalization split is configured in `DataConfig`: `train_years=(2016,)` and `eval_years=(2017,)` for City Hotel by default. Training scripts and learned-policy experiments should train on `load_train_historical_data()` and evaluate on `load_eval_historical_data()`. Baseline search/parameter selection should also use the training split, with reported performance evaluated on the eval split, unless an experiment is explicitly labeled as an oracle eval-set benchmark.

For algorithm experimentation, prefer adding new trainers behind `src/training/algorithm_registry.py` and reusing the shared `train_single_run` / `build_eval_env` interface rather than hard-coding `if algo == ...` branches throughout experiment scripts. Keep experiment outputs algorithm-labeled in run names, CSV rows, and summary files.

For PPO action-distribution experiments, use the independent algorithm names exposed by `src/training/algorithm_registry.py`: `ppo_standard`, `ppo_tanh_gaussian`, `ppo_truncated_gaussian`, `ppo_scale_adjusted_truncated_gaussian`, and `ppo_beta`. The registry wrappers set `PPOConfig.policy_variant` for each run; experiment scripts should not manually mutate `PPO_CONFIG.policy_variant`. Treat `ppo_tanh_gaussian` as the squashed/logit-normal-style bounded policy, `ppo_truncated_gaussian` / `ppo_scale_adjusted_truncated_gaussian` as the truncated-normal alternatives, and `ppo_beta` as the bounded Beta-distribution alternative for diagnosing boundary-action bias.

For baseline experimentation, keep reusable pricing policies and search helpers under `src/baseline/`. Experiment scripts should orchestrate scenarios, outputs, and plots, but should not duplicate baseline policy implementations.

For scenario-specific experiments, keep the canonical scenario list in `configs/scenario_policy_training_scenarios.json`. `experiment_scenario_policy_training.py` and `experiment_penalty_ablation.py` should read this JSON via `--scenario-file` rather than hard-coding scenario parameters. `experiment_scenario_policy_training.py` computes baselines once per scenario, then parallelizes learned-policy training at the `scenario × algo` job level. For scarcity-penalty ablation, keep training to a single `--train-seed` by default and use multiple `--eval-seeds` for robustness unless the user explicitly asks for multiple independent training seeds.

For the benchmark script, note that multi-process runs with `--max-workers > 1` may not show Stable-Baselines3's per-worker progress bars cleanly in the main terminal. In the current multi-algorithm benchmark, each worker computes baseline searches before launching RL training, so visible training progress may appear delayed even when training is running correctly.

For the current `idea2` customer segmentation, treat `flexible_customer_share`, `lambda_day_mismatch_biz`, and `lambda_day_mismatch_flex` as scenario parameters first, not as directly identified facts from the booking data. When studying cross-day substitution, prefer sweeping these configuration values in experiments before adding more ABM complexity.

The current three-state `idea2` demand calibration uses a quantile-compressed lead-time interpretation. `ideal_offset_probs` is calibrated from the relative frequencies of historical `lead_time` 0, 1, and 2 within that short-lead subset. WTP is not calibrated directly from those narrow lead-time buckets; instead, all valid City Hotel observations are sorted by `lead_time` and split according to the same three probabilities. The resulting three ADR distributions calibrate offset-specific WTP for near-, mid-, and far-horizon preference groups. WTP is bound to the customer's original `ideal_offset` and does not change if a `flex` customer shifts to another date.

Current research findings indicate two important cautions:

- Reward penalties alone do not explain the extreme price structures observed in experiments; cross-day substitution in the ABM is currently the dominant driver of "low-price funnel + high-price blocking" behavior.
- `day0` being relatively cheaper is not automatically a bug, because it has the shortest remaining selling horizon. The modeling concern is when this mild dynamic-pricing effect is amplified into extreme cross-day funneling.
- In the segmented-demand version of `idea2`, the key structural question is how much substitution comes from the `flex` segment versus how much rigid demand is anchored by the `biz` segment. Keep that distinction explicit in experiment design and result interpretation.

The environment uses a single reward definition: contribution profit minus weighted scarcity penalty. Set `EnvConfig.scarcity_penalty_coef=0.0` for the no-scarcity-penalty case. Preserve separate logging for contribution-profit revenue, total penalty, and scarcity penalty so experimental comparisons remain interpretable.

`EnvConfig.variable_cost_per_room` controls the per-accepted-room variable cost. Environment `revenue` metrics are contribution-profit metrics when this value is positive: `(price - variable_cost_per_room) * accepted_rooms`. The environment also logs `gross_revenue` and `variable_cost` separately for interpretation.
