"""PPO调参模块（实验二最小方案）。"""

'''
conda run -n abm python -m src.training.tuning.experiment2.tune_ppo \
  --mode debug --coarse-trials 2 --refine-trials 1 \
  --coarse-episodes 100 --refine-episodes 150 --final-episodes 200 \
  --coarse-seeds 1 --refine-seeds 1 --final-seeds 2 --post-eval-episodes 10 \
  --trial-jobs 1 --seed-jobs 1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m src.training.tuning.experiment2.tune_ppo \
  --mode debug --coarse-trials 24 --refine-trials 12 \
  --coarse-episodes 300 --refine-episodes 600 --final-episodes 1000 \
  --coarse-seeds 1 --refine-seeds 3 --final-seeds 5 --post-eval-episodes 30 \
  --trial-jobs 6 --seed-jobs 5


--- 快速
conda run -n abm python -m src.training.tuning.experiment2.tune_ppo \
  --mode debug \
  --coarse-trials 12 \
  --refine-trials 6 \
  --coarse-episodes 100 \
  --refine-episodes 200 \
  --final-episodes 300 \
  --coarse-seeds 1 \
  --refine-seeds 1 \
  --final-seeds 1 \
  --post-eval-episodes 15 \
  --sampler-seed 20260530 \
    --trial-jobs 4 --seed-jobs 1

-- mediun
conda run -n abm python -m src.training.tuning.experiment2.tune_ppo \
  --mode medium \
  --coarse-trials 24 \
  --refine-trials 12 \
  --coarse-episodes 300 \
  --refine-episodes 600 \
  --final-episodes 1000 \
  --coarse-seeds 2 \
  --refine-seeds 3 \
  --final-seeds 5 \
  --post-eval-episodes 30 \
  --sampler-seed 20260530

'''
