# 未实现想法备忘

本文档用于记录当前暂不进入实现范围、但值得在后续研究中保留的模型与实验想法。这里的内容不是当前 `idea2` baseline 的验收范围；除非后续明确决定推进，否则不应直接修改 `src/` 中的环境或训练逻辑。

## Reward 按入住日结算与跨期信用分配

### 背景问题

当前环境的 reward 主要在每日 step 后即时计算：

```text
reward = revenue - full_capacity_penalty - scarcity_penalty
```

这种设计简单、可训练，也便于观察收入、满房惩罚和稀缺惩罚。但它存在一个重要的跨期信用分配问题：同一个入住日的最终销售结果并不是由某一天的价格单独决定的，而是由它在三天滚动窗口中连续被定价的过程共同决定。

例如入住日 `D` 会依次经历：

```text
t = D - 2 时作为 day2 被定价
t = D - 1 时作为 day1 被定价
t = D     时作为 day0 被定价
```

因此，入住日 `D` 的最终收入、是否过早售罄、是否低价消耗库存，都与前两天的价格选择有关。如果只在每天即时给收入和库存惩罚，PPO/SAC 需要通过未来回报间接学习这种因果关系，训练信号会比较噪声。

### 核心想法

可以考虑把 reward 从“每日即时记账”改为“按入住日 cohort 结算”。也就是等某个入住日完整经历三天销售窗口并结束后，再统一评价这个入住日的最终表现。

一个可能的入住日价值定义为：

```text
stay_day_value(D)
  = total_revenue_for_D
  - early_sellout_penalty(D)
  - unsold_inventory_penalty(D)
  - low_price_opportunity_cost(D)
```

其中：

- `total_revenue_for_D`：入住日 `D` 在三天窗口内累计获得的收入。
- `early_sellout_penalty(D)`：如果入住日过早满房，惩罚其丧失后续高价值需求的机会。
- `unsold_inventory_penalty(D)`：如果入住日结束时仍有大量空房，可作为需求不足或定价过高的惩罚。
- `low_price_opportunity_cost(D)`：如果大量房间以明显偏低价格售出，可作为低价消耗稀缺容量的机会成本。

然后把这个入住日最终价值按一定比例分配给三次历史定价动作：

```text
t = D - 2: credit += w2 * stay_day_value(D)
t = D - 1: credit += w1 * stay_day_value(D)
t = D:     credit += w0 * stay_day_value(D)
```

权重可以作为实验参数，例如：

```text
w2 = 0.2, w1 = 0.3, w0 = 0.5
```

或者如果认为提前两天的价格更强地决定需求流入，也可以提高 `w2` 和 `w1` 的权重。

### 实现形式设想

一种直接做法是 delayed reward：

- 环境内部维护每个入住日的销售账本。
- 每天记录该入住日在当前 offset 下的价格、请求量、接受量、收入和剩余库存。
- 当入住日从窗口中滚出或结束时，计算完整的 `stay_day_value`。
- 当前 step 返回这个刚结算入住日的 reward。

这种形式的优点是经济含义比较清楚；缺点是 reward 会更延迟，PPO 训练可能更困难。

另一种折中做法是 hybrid reward：

```text
step_reward
  = immediate_revenue_weight * today_booking_revenue
  + settlement_weight * checkout_settlement_reward
```

也就是保留一部分即时收入信号，同时把入住日最终结算价值作为补充。这样可以降低 reward 稀疏性，又能比当前设计更明确地表达跨期容量机会成本。

### 预期优点

- 更符合三天滚动定价窗口的经济结构。
- 可以减少“今天被惩罚但原因来自前几天动作”的归因噪声。
- 便于解释某个入住日的最终收入、售罄时点和价格路径之间的关系。
- 可以作为对照实验，判断当前极端价格结构到底来自 ABM 的跨日替代，还是来自 reward 设计放大的信用分配偏差。

### 主要风险

- reward 更延迟，PPO/SAC 的训练稳定性可能下降。
- 如果简单把最后一天收入机械分摊给前两天，可能高估早期定价动作的贡献。
- 需要新增入住日级别的账本和日志，环境复杂度会上升。
- 如果同时改 reward 和 PPO 训练细节，会难以判断实验结果变化的来源。

### 当前决定

暂不实现该 reward 修改。

当前优先级仍是先处理 PPO 本身存在的问题，并保留现有 reward 作为 baseline。等 PPO 训练、动作分布、评估流程等基础问题更稳定后，再考虑把该想法实现为可配置的 reward mode，例如：

```text
reward_mode = "daily_state_penalty"   # 当前 baseline
reward_mode = "stay_day_settlement"   # 按入住日结算
reward_mode = "hybrid_settlement"     # 即时收入 + 入住日结算
```

后续如果推进，应先做最小对照实验，而不是直接替换主线 reward。
