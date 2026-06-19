# 第一阶段：环境审计与实验协议

本目录用于完成论文项目第一阶段任务：

- 固化环境诊断流程
- 检查 ABM 环境是否支持“动态定价值得存在”
- 输出固定价格扫描结果
- 导出当前运行配置快照
- 形成可复现的实验协议

## 目录说明

- `run_fixed_price_scan.py`
  - 对一组固定价格策略进行扫描，输出酒店收益、OTA 收益、系统收益、在线占比等指标。
- `run_environment_diagnostics.py`
  - 输出数据集摘要、到达率离散度、ABM/实验二配置快照等诊断文件。
- `artifacts/`
  - 运行脚本后自动生成的结果目录。

## 推荐执行顺序

1. 运行环境诊断
2. 运行固定价格扫描
3. 基于结果判断：
   - 简单低价是否异常强
   - 库存约束是否足够紧
   - 需求波动是否合理
   - 当前 ABM 是否支持动态定价研究命题

## 命令示例

```bash
conda run -n abm python phase1_environment_audit/run_environment_diagnostics.py --mode debug
```

```bash
conda run -n abm python phase1_environment_audit/run_fixed_price_scan.py --mode debug --price-min 60 --price-max 160 --price-step 10
```

## 关键输出文件

- `artifacts/config_snapshots/runtime_config_snapshot.json`
- `artifacts/diagnostics/city_hotel_dataset_summary.json`
- `artifacts/diagnostics/arrival_rate_month_daytype.csv`
- `artifacts/fixed_price_scan/fixed_price_scan_episode.csv`
- `artifacts/fixed_price_scan/fixed_price_scan_summary.csv`

## 阶段验收标准

- 能清楚回答“固定低价是否已经足够强”
- 能输出当前 ABM 与实验二的配置快照
- 能复现实验协议，不依赖手工口头说明
