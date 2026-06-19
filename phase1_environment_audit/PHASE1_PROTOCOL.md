# 第一阶段协议

## 目标

第一阶段只回答三个问题：

1. 当前 ABM 环境是否足够可信，能够支持动态定价研究命题。
2. 简单静态价格策略是否已经足够强，从而挤压了动态学习方法的价值空间。
3. 实验二后续比较时，应该采用怎样的环境与评估协议。

## 必做任务

### 1. 环境诊断

运行：

```bash
conda run -n abm python phase1_environment_audit/run_environment_diagnostics.py --mode debug
```

检查输出：

- `city_hotel_dataset_summary.json`
- `lead_time_summary.json`
- `wtp_summary.json`
- `arrival_rate_month_daytype.csv`
- `runtime_config_snapshot.json`

核心检查点：

- 取消率是否过高，是否会影响到达率口径解释。
- lead time 条件分布是否已经成功构建。
- WTP 分布是否仍然过于依赖 ADR。
- 工作日/节假日到达率差异是否符合 City Hotel 的业务直觉。

### 2. 固定价格扫描

运行：

```bash
conda run -n abm python phase1_environment_audit/run_fixed_price_scan.py --mode debug --price-min 80 --price-max 120 --price-step 5
```

检查输出：

- `fixed_price_scan_episode.csv`
- `fixed_price_scan_summary.csv`

核心检查点：

- 最优固定价是否总是落在最低价附近。
- 酒店收益是否随着价格上升单调下降。
- 在线份额和入住率是否出现异常形态。

## 结果解释规则

### 情况 A：最低价几乎总是最优

说明：

- 当前环境很可能对低价过度友好。
- 要优先检查库存容量、需求价格弹性、取消机制、WTP 分布和 reward 设计。

结论：

- 不应直接进入“算法强弱”结论阶段。

### 情况 B：存在清晰的中间最优价

说明：

- 环境具备合理的价格-收益权衡。
- 动态定价问题有继续研究的价值。

结论：

- 可以进入第二阶段，继续稳定 CEM 和比较 baseline。

### 情况 C：不同场景下最优固定价变化很大

说明：

- 环境具备非平稳特征。
- OOD/扰动实验对论文将非常重要。

## 第一阶段完成标准

- 输出完整配置快照
- 输出固定价格扫描结果
- 对“低价策略是否异常强”给出明确判断
- 对第二阶段是否继续以当前环境为主环境给出决策
