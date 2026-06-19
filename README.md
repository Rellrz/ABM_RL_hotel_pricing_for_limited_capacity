# 酒店-OTA动态定价（ABM + CEM/MARL）

本项目用于研究酒店在 OTA 渠道下的动态定价问题，核心由三部分组成：
- `ABM`：消费者到达、支付意愿、渠道选择与库存约束仿真。
- `决策策略`：酒店（CEM/其他）与 OTA 启发式补贴交互。
- `实验框架`：支持实验一、实验二、对比消融、调参与结果分析。

设计文档：
- [基于ABM-MARL的酒店-OTA动态定价博弈系统设计文档.md](docs/%E5%9F%BA%E4%BA%8EABM-MARL%E7%9A%84%E9%85%92%E5%BA%97-OTA%E5%8A%A8%E6%80%81%E5%AE%9A%E4%BB%B7%E5%8D%9A%E5%BC%88%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md)
- [基于ABM-MARL的酒店-OTA动态定价博弈系统设计文档.pdf](docs/基于ABM-MARL的酒店-OTA动态定价博弈系统设计文档.pdf)

## 快速开始
```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 跑实验二（debug）
python experiments/experiment2.py --mode debug
```

## 主要目录
```text
ABM_hotel_pricing/
├── configs/                         # 配置系统（已拆分）
│   ├── config.py                    # 兼容入口（旧导入路径仍可用）
│   ├── schema.py                    # dataclass 定义
│   ├── estimators.py                # 从数据估计 ABM 参数
│   ├── defaults.py                  # 扰动模板与默认参数
│   ├── loader.py                    # 运行时装配
│   └── validators.py                # 配置校验
├── src/
│   ├── environment/                 # ABM + 环境封装
│   ├── agent/                       # 酒店/OTA智能体
│   ├── algorithms/                  # CEM/CEM-NN/Multivariate CEM
│   ├── training/                    # 主训练器与实验runner
│   ├── evaluation/                  # 评估与统计
│   ├── plot/                        # 绘图
│   └── utils/                       # 通用工具（分桶、状态、奖励）
├── experiments/
│   ├── train_game.py                # 主线训练入口
│   └── experiment2.py               # 实验二（对比/消融）入口
└── outputs/                         # 模型、日志、图表与实验结果
```

## 配置与扰动模板

当前不再通过环境变量切换扰动。  
请直接修改 `configs/config.py` 中这一行：

```python
ABM_PERTURBATION_TEMPLATE = 'none'  # none / mild / medium / stress
```

含义：
- `none`：无扰动（ID）
- `mild`：轻度扰动
- `medium`：中度扰动
- `stress`：压力测试（强扰动）

## 运行命令

### 1) 主线训练
```bash
python experiments/train_game.py \
  --episodes 400 \
  --mode simultaneous \
  --commission 0.20 \
  --subsidy-ratio-max 0.8 \
  --update-frequency 30 \
  --booking-window-days 91 \
  --decision-buckets "0|1|2-3|4-6|7-13|14-29|30-59|60-90"
```

### 2) 实验二：对比与消融
```bash
# debug
python experiments/experiment2.py --mode debug

# medium/full
python experiments/experiment2.py --mode medium --n-jobs 4
python experiments/experiment2.py --mode full --n-jobs 8
```

常用选项：
- `--skip-ppo`
- `--skip-qlearning`
- `--skip-cem`
- `--skip-emsrb`

## 结果输出

### 实验二结果
- `outputs/experiment2/results/experiment2_training.csv`
- `outputs/experiment2/results/experiment2_post_eval.csv`
- `outputs/experiment2/results/performance_table_*.csv`
- `outputs/experiment2/results/experiment2_stats_*.csv`
- `outputs/experiment2/results/experiment2_summary.json`

### 实验二图表
- `outputs/experiment2/figures/episode_revenue_curves_hotel.pdf`
- `outputs/experiment2/figures/episode_profit_curves_ota.pdf`
- `outputs/experiment2/figures/episode_total_profit_curves_system.pdf`
- `outputs/experiment2/figures/post_eval_bar_*_with_errorbars.pdf`

### 训练模型与日志
- `outputs/models/*.json`
- `outputs/models/training_data_*.csv`
- `outputs/tensorboard_logs/`

### 主线训练说明
- `hotel_env` 负责输出原始状态；状态补齐、分桶映射和奖励计算统一放在 `src/utils/common.py`
- `CEM` 当前使用更丰富的状态键：`(stage_id, season, weekday, bucket_inv_bin, near_inv_bin, far_inv_bin, inv_slope_bin)`
- `Q-learning` 仍保留 `240` 个离散状态（`5 x 3 x 2 x 8`）
- 当前主线奖励为“酒店收益主目标 + 轻量机会成本惩罚”，训练日志中可同时看到 `TrainBase / TrainShaped / ShapePenalty`

TensorBoard：
```bash
tensorboard --logdir=outputs/tensorboard_logs
```

## 当前ABM关键机制（简述）
- `lead_time`：经验分布采样，并支持条件分布（season × weekday）。
- `WTP`：基于历史 ADR 分层拟合并采样。
- `customer_type`：`online_only / omnichannel` 分群，再映射到最终成交渠道。
- `扰动`：支持需求 OU+跳跃、WTP 漂移、渠道偏好扰动、效用噪声。

## 开发说明
- 配置拆分后，建议新代码使用 `from configs import ...` 或 `from configs.schema import ...`
- 旧代码仍可使用 `from configs.config import ...`（兼容层保留）
- 实验二不再维护独立实现目录；`experiments/experiment2.py` 直接调用 `src/` 下模块
- 与状态/奖励相关的改动优先集中到 `src/utils/common.py`

## 许可证
如需开源发布，请在仓库补充 `LICENSE` 文件并在本 README 标注许可类型。
