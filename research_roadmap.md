# idea2 研究路线与投稿规划

本文档用于沉淀当前 `idea2` 酒店短订窗口动态定价研究的具体步骤、实验流程与投稿方向。当前研究主线不是证明某个算法天然优越，而是回答：

```text
在短订窗口酒店定价中，当消费者存在理想入住日、跨日替代、参考价格心理和容量约束时，
动态定价策略是否能利用库存状态与需求结构获得超过强静态定价的收益？
```

因此，研究流程应遵循：

```text
先验证环境机制是否产生动态定价空间
再在有动态空间的场景训练 RL
最后解释策略行为并做稳健性检验
```

## 1. 固定当前 idea2 baseline

### 目标

明确当前研究边界，避免后续实验解释混乱。

### 当前模型边界

- 三天滚动定价窗口。
- 酒店每天同时设置 `day0/day1/day2` 三个价格。
- 消费者分为 `biz` 和 `flex` 两类。
- `biz` 消费者基本锚定理想入住日。
- `flex` 消费者可在三天窗口中根据价格跨日替代。
- `flexible_customer_share` 是机制场景参数，不直接声称由数据识别。
- `lambda_day_mismatch_flex` 是 flex 消费者跨日替代强度参数。
- 参考价格机制保留。
- 容量约束为硬约束，不允许超售。
- 当前 reward 暂时保持不变。

### 产出

- 模型设定说明。
- 参数表。
- 代码结构说明。
- baseline 策略定义。

## 2. 强 baseline 诊断

### 目标

先判断环境是否真的存在动态定价空间。如果强静态策略已经接近最优，则 RL 不容易稳定超过静态价格。

### 已实现 baseline

当前 reusable baseline 策略已放入：

```text
src/baseline/pricing_baselines.py
```

包括：

- `static_grid_best`
- `weekday_weekend_static_best`
- `inventory_protection_best`

### 策略含义

`static_grid_best`：

```text
全 episode 使用同一组三天价格 [p0, p1, p2]
```

`weekday_weekend_static_best`：

```text
工作日使用一组三天价格
周末使用另一组三天价格
```

`inventory_protection_best`：

```text
price_tau = base_price_tau + alpha * (1 - inventory_tau / capacity)
```

即库存越紧，价格越高。

### 已得到的关键结论

当前诊断结果显示：

- 全局静态三价策略已经很强。
- 工作日/周末分价相对全局静态提升很小。
- 库存保护策略只在低容量场景 `capacity=20` 明显超过静态价格。

这说明当前 `idea2` 环境里的动态定价价值主要来自容量紧张，而不是简单周度差异。

### 产出

- `dynamic_baseline_diagnostics` 实验结果。
- 三类 baseline 的 revenue、reward、full rate、价格结构对比。
- 证明动态空间主要集中在低容量场景。

## 3. 机制强度实验

### 目标

回答 `idea2` 的核心行为问题：

```text
跨日替代机制什么时候会创造动态定价价值？
```

### 第一轮机制网格

已执行的正式机制诊断实验：

```text
flexible_customer_share = 0.0, 0.25, 0.5, 0.75, 1.0
lambda_day_mismatch_flex = 6, 12, 24, 48
capacity = 20, 30, 50
```

该实验不训练 RL，只比较：

```text
static_grid_best
weekday_weekend_static_best
inventory_protection_best
```

### 当前结果解释

按容量聚合：

```text
capacity=20:
inventory_protection_vs_static 平均约 1.05，最高约 1.085

capacity=30:
inventory_protection_vs_static 平均约 1.003，最高约 1.020

capacity=50:
inventory_protection_vs_static 基本为 1.000
weekday_weekend_vs_static 最高约 1.013
```

最值得后续训练 SAC 的强动态空间场景：

```text
capacity=20, flexible_customer_share=0.50, lambda_day_mismatch_flex=48
capacity=20, flexible_customer_share=0.75, lambda_day_mismatch_flex=48
capacity=20, flexible_customer_share=1.00, lambda_day_mismatch_flex=48
```

### 产出

- 机制热力图。
- `weekday_weekend_vs_static` 与 `inventory_protection_vs_static` 对比。
- 筛选后续 RL 训练场景。

## 4. 在有动态空间的场景训练 SAC

### 目标

只在 baseline 诊断证明存在动态空间的场景训练 RL，避免在动态价值很弱的场景浪费训练成本。

### 主算法

第一阶段主算法建议使用 SAC。

原因：

- 当前 benchmark 中 SAC 明显优于 PPO。
- SAC 更适合连续动作定价。
- SAC 的 off-policy replay 对随机 ABM 环境更友好。
- PPO 当前容易学出边界化价格或低价填房策略。

### 第一批 SAC 场景

```text
capacity=20, flexible_customer_share=0.50, lambda_day_mismatch_flex=48
capacity=20, flexible_customer_share=0.75, lambda_day_mismatch_flex=48
capacity=20, flexible_customer_share=1.00, lambda_day_mismatch_flex=48
```

### 对照策略

每个场景至少比较：

```text
SAC
static_grid_best
weekday_weekend_static_best
inventory_protection_best
```

### 关键判断

如果：

```text
SAC >= static_grid_best
SAC < inventory_protection_best
```

说明 SAC 学到部分动态价值，但还不如手工库存保护规则。

如果：

```text
SAC >= inventory_protection_best
```

说明 RL 能捕捉库存保护规律，并可能学到更复杂的状态依赖策略。

如果：

```text
SAC < static_grid_best
```

说明算法训练、状态表示或 reward 信号仍有问题。

### 建议新增实验文件

```text
experiments/experiment_sac_selected_mechanisms.py
```

该文件只做一件事：

```text
对机制诊断筛出来的场景训练 SAC，并和 static / weekday_weekend / inventory baseline 比较。
```

## 5. SAC 策略解释

### 目标

不要只报告 revenue，而要解释 SAC 学到的价格行为是否有经济意义。

### 必须输出的指标

```text
episode_revenue
episode_reward
episode_penalty
episode_acceptance_rate
avg_price_day0/day1/day2
avg_inventory_day0/day1/day2
full_rate_day0/day1/day2
revenue_per_capacity_day
```

### 策略切片

建议在训练完成后做以下切片：

```text
同一 weekday 下，库存高/中/低时的价格
同一库存水平下，day0/day1/day2 的价格
参考价格高/低时的价格响应
不同 flex share 下的价格结构变化
```

### 要回答的问题

- SAC 是否真的根据库存涨价？
- SAC 是否只是学到近似静态价格？
- SAC 是否制造 low-price funnel？
- SAC 是否过度追求满房？
- SAC 的价格结构是否优于 inventory protection baseline？

## 6. PPO 作为算法鲁棒性对照

### 定位

PPO 不再作为主线算法，但可作为算法适配性对照。

### 建议只保留有限实验

```text
PPO-standard, n_steps=512
PPO-standard, n_steps=1024
```

避免无限调 PPO。

### 可能形成的结论

```text
on-policy PPO 在随机 ABM + 连续动作 + 跨期容量约束环境中样本效率较弱，
更容易学到边界化价格或低价填房策略；
SAC 更适合作为该环境下的主学习算法。
```

## 7. 稳健性实验

### 优先级

在 SAC 主线跑通后，再做稳健性实验。

建议顺序：

```text
1. 不同 eval seeds
2. 不同 capacity
3. 不同 flexible_customer_share
4. reference price ablation
5. penalty ablation
6. flex segment ablation
```

### 重点 ablation

参考价格消融：

```text
lambda_reference_price = 0
vs
lambda_reference_price = 当前值
```

惩罚消融：

```text
with penalty
vs
no penalty
```

flex segment 消融：

```text
flexible_customer_share = 0
vs
flexible_customer_share = 1
```

### 目标

解释极端价格结构来自哪里：

- 跨日替代？
- 参考价格心理？
- 容量 penalty？
- 算法偏差？

## 8. Reward settlement 暂作 future work

之前提出的按入住日 cohort 结算 reward 很有价值，但不建议进入当前主线。

原因：

- 当前更核心的问题是动态空间和算法适配。
- 一旦同时改 reward 和机制参数，很难判断结果变化来自哪里。
- 现有 SAC / baseline 诊断已经能支撑主要研究结论。

建议放入：

```text
future work
或 robustness extension
```

## 9. 推荐论文结构

```text
1. Introduction
   - 短订窗口酒店动态定价问题
   - 为什么传统静态/解析需求方法不足
   - 本文研究问题

2. Model
   - 三天滚动定价窗口
   - biz/flex 消费者 ABM
   - 参考价格心理
   - 容量约束与 reward

3. Baselines and Diagnostics
   - static_grid_best
   - weekday_weekend_static_best
   - inventory_protection_best
   - 先证明动态空间在哪里

4. Mechanism Experiments
   - flexible_customer_share sweep
   - lambda_day_mismatch_flex sweep
   - capacity sweep

5. RL Experiments
   - SAC selected mechanism scenarios
   - PPO as robustness comparison

6. Policy Interpretation
   - 库存-价格响应
   - day0/day1/day2 价格结构
   - full rate 与 revenue trade-off

7. Robustness and Ablation
   - reference price
   - penalty
   - flex segment

8. Conclusion
   - 动态价值主要来自容量紧张和跨日替代结构
   - SAC 在有动态空间的场景中是否能捕捉库存保护逻辑
   - 局限与 future work
```

## 10. 目标期刊推荐

以下推荐按“论文最终强调点”分层，而不是只按期刊名排序。投稿前应再次检查最新 aims and scope、近期文章风格和格式要求。

### 第一梯队：最贴合收入管理与定价

#### Journal of Revenue and Pricing Management

官网：

```text
https://link.springer.com/journal/41272/aims-and-scope
```

适配理由：

- 期刊核心就是 revenue management 和 pricing。
- 官方 scope 明确覆盖 pricing、segmentation、capacity and inventory allocation、demand modelling 等主题。
- 本研究的酒店容量约束、动态定价、库存保护 baseline、SAC 定价策略都非常贴合。

建议定位：

```text
首选目标。
强调 revenue management / pricing / capacity control / simulation-based dynamic pricing。
```

投稿叙事重点：

- 强 baseline 诊断。
- 何时动态定价能超过静态定价。
- SAC 是否能学习库存保护。
- 管理启示要写得清楚。

### 第二梯队：酒店管理与运营管理导向

#### International Journal of Contemporary Hospitality Management

官网：

```text
https://www.emeraldgrouppublishing.com/journal/ijchm
```

适配理由：

- 官方说明该刊关注 hospitality and tourism businesses 的管理问题。
- 覆盖 strategic management、operations、marketing、finance、HR 等 hospitality management 议题。
- 如果本文强调酒店收益管理、运营决策、管理启示，而不是算法本身，则较适合。

建议定位：

```text
酒店管理导向目标。
需要加强 theoretical contribution 和 managerial implications。
```

投稿叙事重点：

- 酒店短订窗口的管理问题。
- 刚性/灵活需求分层对定价策略的影响。
- 容量紧张下动态定价何时有管理价值。
- 不要写成纯算法论文。

#### Tourism Management

官网：

```text
https://www.sciencedirect.com/journal/tourism-management/about/aims-and-scope
```

适配理由：

- 官方 scope 强调 tourism management 的管理、规划、政策问题，以及理论或方法贡献。
- 如果本文能从酒店短订窗口扩展到旅游住宿需求管理，并突出理论/方法创新，可作为较高目标。

建议定位：

```text
高目标期刊。
需要显著强化理论贡献、方法贡献和旅游管理含义。
```

投稿叙事重点：

- 不只是酒店定价实验，而是旅游住宿需求管理中的行为机制与动态决策问题。
- 需要把 ABM + RL 的方法贡献讲清楚。
- 需要避免只呈现工程仿真结果。

### 第三梯队：决策支持与分析系统导向

#### Decision Support Systems

官网：

```text
https://www.sciencedirect.com/journal/decision-support-systems/about/aims-and-scope
```

适配理由：

- 官方 scope 关注 enhanced decision making 的理论与技术问题。
- 本研究如果发展成一个面向酒店收益管理的仿真-学习决策支持框架，可考虑该方向。

建议定位：

```text
方法与决策支持导向目标。
需要把贡献从“酒店实验”提升为“动态定价决策支持系统”。
```

投稿叙事重点：

- ABM 环境作为决策支持仿真器。
- RL 与强 baseline 的可解释比较。
- 策略评估、场景诊断、管理决策支持流程。

### 备选方向

如果后续论文更偏旅游/酒店实证与管理启示，可考虑：

```text
International Journal of Hospitality Management
Journal of Hospitality & Tourism Research
Tourism Management Perspectives
```

如果后续论文更偏算法和运筹优化，需要更强的方法创新，目前的 model-free RL + ABM 可能还不够，需要增加：

- 更强的算法贡献。
- 更严格的最优性或 regret 对照。
- 更大规模 benchmark。
- 与经典 revenue management 方法的系统比较。

## 11. 当前最推荐投稿路线

基于当前项目状态，最稳妥的投稿路线是：

```text
主目标：Journal of Revenue and Pricing Management
备选管理类目标：International Journal of Contemporary Hospitality Management
高目标尝试：Tourism Management
方法系统方向备选：Decision Support Systems
```

当前最应该补强的是：

```text
1. SAC selected mechanisms 实验
2. 策略解释图
3. 与 inventory_protection_best 的深入比较
4. reference price / penalty / flex segment 消融
```

当这些结果完整后，文章会更像一篇有清晰管理问题和实验发现的 revenue management 论文，而不是单纯的 RL 调参报告。

## 12. 已查阅的期刊官方页面

- Journal of Revenue and Pricing Management: https://link.springer.com/journal/41272/aims-and-scope
- International Journal of Contemporary Hospitality Management: https://www.emeraldgrouppublishing.com/journal/ijchm
- Tourism Management: https://www.sciencedirect.com/journal/tourism-management/about/aims-and-scope
- Decision Support Systems: https://www.sciencedirect.com/journal/decision-support-systems/about/aims-and-scope
