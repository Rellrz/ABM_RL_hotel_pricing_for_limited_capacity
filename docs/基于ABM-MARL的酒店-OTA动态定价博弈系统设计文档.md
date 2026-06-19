
> **算法基础**: Cross-Entropy Method (CEM)
>
> **系统架构**: 基于智能体建模（ABM）的多智能体强化学习（MARL）博弈框架
>
> **核心场景**: 酒店双渠道动态定价与OTA启发式补贴联动

---

## 目录

[[#1. 系统概述]]
[[#2. 项目架构与模块组织]]
[[#3. ABM环境建模]]
[[#4. 酒店Agent设计]]
[[#5. OTA Agent设计]]
[[#6. Agent与环境的交互机制]]
[[#7. 奖励函数设计]]
[[#8. 伪代码]]
[[#9. 数据来源与预处理]]
[[#10. 实验配置与超参数]]
[[#11. Cross-Entropy Method算法实现（酒店）]]
[[#12. 关键设计决策总结]]

---

## 1. 系统概述

### 1.1 研究背景

在现代酒店行业中，酒店通常同时通过线上（OTA渠道，如携程、美团等）和线下（前台直销）两种渠道进行客房销售。酒店需要在两个渠道上分别制定价格策略，而OTA平台则通过佣金收入和补贴策略来影响线上渠道的最终售价，以此吸引更多客户通过OTA渠道预订。

这构成了一个典型的**双层博弈问题**：酒店（领导者）制定线上基础价格和线下价格，OTA（跟随者）基于酒店的定价决定补贴比例。双方的策略相互影响，最终通过消费者的预订行为体现为各自的收益。

### 1.2 系统目标

本系统旨在构建一个完整的仿真与学习框架，实现以下目标：

1. **真实市场模拟**：通过ABM（Agent-Based Model）方法，基于历史酒店预订数据模拟消费者到达、决策与预订行为，还原真实需求响应。
2. **酒店策略学习**：酒店Agent使用CEM/CEM-NN进行价格学习，其中 `cem` 模式已升级为联合决策的多元高斯CEM。
3. **OTA外生响应**：OTA不再学习，改为“时间敏感 + 随机扰动”的启发式补贴策略，提升稳定性与可解释性。
4. **策略联动分析**：在“酒店学习 + OTA反应式规则”框架下分析价格、渠道分配与收益变化。

### 1.3 技术架构总览

系统采用三层架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                     训练控制层                            │
│  GameTrainer: 管理训练流程、策略更新、数据记录              │
├─────────────────────────────────────────────────────────┤
│                     决策层（MARL）                        │
│  ┌──────────────────┐    ┌───────────────────┐          │
│  │  HotelAgent      │    │   OTA Heuristic   │          │
│  │  (多元CEM/CEM-NN) │    │   (外生反应规则)    │          │
│  │  输出:线上基础价格  │    │   输出:补贴比例      │          │
│  │      线下价格      │    │   (0%-80%)       │          │
│  └───────┬──────────┘    └────────┬──────────┘          │
│          │         策略交互        │                     │
│          └───────────┬────────────┘                     │
├──────────────────────┼──────────────────────────────────┤
│                      ▼                                  │
│                  环境层（ABM）                            │
│  ┌──────────────────────────────────────────────┐       │
│  │  HotelEnvironment                            │       │
│  │  ├── 状态管理（库存、季节、日期类型）           │       │
│  │  ├── 价格窗口管理（91天滚动窗口）              │       │
│  │  └── HotelABMModel（Mesa/兼容降级实现）        │       │
│  │       ├── 消费者生成（泊松过程）               │       │
│  │       ├── 消费者决策（效用函数）               │       │
│  │       └── 库存约束与预订处理                   │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 项目架构与模块组织

### 2.1 目录结构

```
ABM_hotel_pricing/
├── configs/
│   ├── config.py                        # 兼容入口
│   ├── schema.py                        # dataclass 配置定义
│   ├── loader.py                        # 配置装配
│   └── defaults.py / validators.py      # 默认值与校验
├── src/
│   ├── algorithms/                      # RL算法实现
│   │   ├── base_algorithm.py            # 算法抽象基类
│   │   ├── cem.py                       # 一维CEM（兼容保留，用于其他模块）
│   │   ├── cem_nn.py                    # 神经网络版CEM算法
│   │   └── multivariate_cem.py          # 多元高斯CEM（酒店联合定价核心）
│   ├── agent/                           # 智能体实现
│   │   ├── hotel_agent_dual_channel.py  # 双渠道酒店Agent
│   │   └── ota_agent.py                 # OTA启发式补贴策略
│   ├── environment/                     # 环境模拟
│   │   ├── hotel_env.py                 # 酒店RL环境
│   │   ├── abm_customer_model.py        # ABM消费者行为模型
│   │   └── bucket_pricing_simulator.py  # 实验二分桶仿真封装
│   ├── training/
│   │   ├── game_trainer.py              # 博弈训练器（酒店学习 + OTA外生响应）
│   │   └── cem_ablation_runner.py       # 实验二 CEM runner
│   ├── evaluation/                      # 评估与统计
│   ├── plot/                            # 绘图
│   └── utils/
│       ├── common.py                    # 分桶、状态补齐、奖励计算
│       └── training_monitor.py          # 训练监控与可视化
├── experiments/
│   ├── train_game.py                    # 主线训练入口
│   └── experiment2.py                   # 实验二入口
├── datasets/
│   └── hotel_bookings.csv               # 历史预订数据
└── outputs/
    ├── models/                          # 训练后的模型参数（JSON）
│   ├── results/                         # 主线训练统计（CSV）
│   ├── experiment2/                     # 实验二结果与图表
    ├── figures/                         # 训练结果可视化图表
    └── tensorboard_logs/                # TensorBoard训练日志
```

### 2.2 核心模块依赖关系

```
train_game.py / experiment2.py（入口）
  ├── game_trainer.py / experiment2 runners（训练流程编排）
  ├── hotel_env.py（原始环境状态）
  │     └── abm_customer_model.py（消费者行为模拟）
  ├── common.py（分桶、状态补齐、奖励计算）
  ├── hotel_agent_dual_channel.py（酒店决策Agent）
  │     └── multivariate_cem.py / cem_nn.py（酒店算法实现）
  └── ota_agent.py（OTA启发式策略）
```

---

## 3. ABM环境建模

ABM（Agent-Based Model）是本系统的核心环境组件，基于Mesa框架实现，负责模拟消费者的到达、特征生成、预订决策等行为，为强化学习Agent提供真实的市场反馈信号。

### 3.1 消费者生成机制

#### 3.1.1 每日消费者到达（泊松过程）

每日潜在消费者数量服从泊松分布，到达率 $\lambda_m$ 按月份从历史数据中估计：

$$N_{\text{day}} \sim \text{Poisson}(\lambda_m)$$

其中 $m$ 为当前月份（1-12）。月份由仿真天数简化计算：$m = (\lfloor \text{day} / 30 \rfloor \mod 12) + 1$。

到达率 $\lambda_m$ 的计算方法：基于 `hotel_bookings.csv` 中各月份的预订记录总数除以30（简化的月天数），得到日均到达率。

```python
# 具体实现（config.py）
monthly_rates[month] = monthly_counts[month] / 30.0
```

#### 3.1.2 消费者特征生成

每个消费者 $i$ 被赋予如下特征向量 $(\text{lead\_time}_i, T_{\text{stay},i}, \text{WTP}_i, \beta_i, \text{type}_i)$：

**1) 提前预订期（Lead Time）$L_i$**

提前预订期从历史数据拟合的经验分布中采样。系统首先从`hotel_bookings.csv` 中统计各提前天数（0-90天）对应的预订频次，构建离散概率分布：

$$L_i \sim \text{Empirical}(p_0, p_1, \ldots, p_{90})$$

其中 $p_d = \frac{\text{count}(L = d)}{\sum_{d'=0}^{90} \text{count}(L = d')}$。

采样后对结果进行裁剪，确保 $L_i \in [0, \text{booking\_window\_days} - 1]$，即最多提前90天预订。

**2) 目标入住日期 $T_{\text{stay},i}$**

$$T_{\text{stay},i} = \text{current\_day} + L_i$$

消费者的目标入住日期等于当前仿真日期加上其提前预订期。

**3) 最高支付意愿（Willingness To Pay, WTP）$\text{WTP}_i$**

支付意愿服从正态分布，参数由`hotel_bookings.csv` 中未取消订单的平均日房价（ADR）拟合：

$$\text{WTP}_i \sim \mathcal{N}(\mu_{\text{adr}}, \sigma_{\text{adr}})$$

拟合时过滤掉已取消订单和异常值（$\text{ADR} \leq 0$ 或 $\text{ADR} \geq 500$），并对采样结果设置下限 $\text{WTP}_i \geq 10.0$。

**4) 消费者类型 $\text{type}_i$**

$$\text{type}_i \sim \text{Categorical}(\text{ota}: 0.3, \text{ota+direct}: 0.7)$$

即30%的消费者仅通过OTA线上渠道预订（`ota`类型），70%的消费者可同时看到线上和线下价格并选择效用更高的渠道（`ota+direct`类型）。

> **关键设计说明**：`ota` 类型消费者只会比较线上渠道价格；`ota+direct` 类型消费者会同时评估线上和线下两个渠道的效用，选择效用更高的渠道进行预订。

### 3.2 消费者预订决策模型

消费者的预订决策基于**效用函数**，核心思想是：消费者评估预订行为的"值得程度"，当效用超过阈值时执行预订。

#### 3.2.1 效用函数

消费者 $i$ 面对价格 $P$ 时的预订效用为：

$$U_i(P) = (\text{WTP}_i \cdot \delta - P) + \frac{\gamma}{L_i + 1}$$

其中各项含义如下：

| 符号 | 含义 | 说明 |
|------|------|------|
| $\text{WTP}_i$ | 最高支付意愿 | 消费者愿意为一晚住宿支付的最高价格 |
| $\delta$ | 渠道折扣系数 | 线上渠道 $\delta = 0.8$（消费者认为线上价值打8折），线下渠道 $\delta = 1.0$ |
| $P$ | 渠道报价 | 线上为OTA最终价格，线下为酒店直销价格 |
| $\gamma$ | 紧迫性权重 | 默认值20，控制提前期对预订决策的影响强度 |
| $L_i$ | 提前预订天数 | 距离入住日还有多少天 |

**效用函数的经济学解读**：

- **第一项**$(\text{WTP}_i \cdot \delta - P)$：经济盈余，反映消费者从预订中获得的净"赚到"感。当WTP远高于价格时，消费者倾向预订。
- **第二项**$\frac{\gamma}{L_i + 1}$：紧迫性效用，提前期越短，入住需求越紧迫，预订意愿越强。

#### 3.2.2 渠道选择逻辑

消费者的渠道选择取决于其类型：

- **`ota` 型消费者**：仅计算线上效用 $U_{\text{online}} = (\text{WTP}_i \times 0.8 - P_{\text{online}})+ \frac{\gamma}{L_i + 1}$
- **`ota+direct` 型消费者**：同时计算两个渠道的效用，选择效用更高的渠道：
  - $U_{\text{online}} = (\text{WTP}_i \times 0.8 - P_{\text{online}}) + \frac{\gamma}{L_i + 1}$
  - $U_{\text{offline}} = (\text{WTP}_i \times 1.0 - P_{\text{offline}}) + \frac{\gamma}{L_i + 1}$
  - 最终选择 $\max(U_{\text{online}}, U_{\text{offline}})$ 对应的渠道

#### 3.2.3 预订决策规则

$$\text{Book} = \begin{cases} \text{True}, & \text{if } U^* > \theta \\ \text{False}, & \text{otherwise} \end{cases}$$

其中 $U^*$ 为选定渠道的效用值，$\theta = -15$ 为预订阈值。阈值为负数意味着即使消费者觉得价格略高于自己的心理价位，在紧迫性足够高时仍可能做出预订。

### 3.3 库存约束

即使消费者决定预订，预订能否成功还取决于目标入住日期的剩余库存：

$$\text{Booking Success} = \begin{cases} \text{True}, & \text{if } \text{available\_rooms}[T_{\text{stay},i}] > 0 \\ \text{False}, & \text{otherwise} \end{cases}$$

成功预订后，对应日期的可用库存减1。系统维护一个以入住日期为键的库存字典 `daily_available_rooms`，确保每个入住日期的库存约束独立管理。

### 3.4 预订记录

每笔成功的预订被记录为 `BookingRecord`，包含以下信息：

| 字段                  | 类型    | 说明                          |
| ------------------- | ----- | --------------------------- |
| `customer_id`       | int   | 消费者唯一ID                     |
| `booking_date`      | int   | 预订发生的仿真日期                   |
| `target_date`       | int   | 目标入住日期                      |
| `paid_price`        | float | 成交价格                        |
| `wtp`               | float | 消费者WTP                      |
| `customer_type`     | str   | 最终选择的渠道（`online`/`offline`） |

### 3.5 每日模拟流程

ABM模型的 `simulate_day()` 方法执行一天的完整模拟：

```
输入: 线上价格窗口, 线下价格窗口, 各日库存
输出: 当日预订统计（按渠道、按提前天数分组）

1. 调用 generate_daily_customers() 生成当日潜在消费者
2. 对每个消费者:
   a. 计算其目标入住日期对应的提前天数 days_ahead
   b. 从价格窗口中获取该提前天数对应的价格
   c. 调用 make_booking_decision() 进行决策
   d. 若决定预订且库存充足，则记录预订并扣减库存
3. 汇总当日统计:
   - 线上/线下新增预订量
   - 按 day_offset 分组的预订量与收入
   - 当日总收入
```

---


## 4. 酒店Agent设计

### 4.1 设计理念

酒店Agent（`HotelAgentDualChannel`）是一个**双渠道联合定价决策器**。在 `cem` 模式下，采用单体多元高斯CEM一次性输出二维动作 `[P_online_base, P_offline]`；在 `cem_nn` 模式下保留神经网络版双头实现。

设计核心考量：
- **渠道联动性**：线上与线下价格存在交叉弹性，需要通过协方差学习联合变化
- **佣金意识**：线上基础价格需覆盖OTA佣金成本
- **外生响应适配**：OTA采用启发式补贴，酒店需学习在该反应机制下的稳健定价

### 4.2 状态空间设计

当前代码将“环境原始状态”和“策略使用状态”分开管理：

- `hotel_env.py` 只输出原始状态，如 `inventory_raw / future_inventory / day / day_offset`
- `src/utils/common.py` 负责状态补齐与离散化
- `Q-learning` 保留标量离散状态
- `CEM` 使用更丰富的 tuple 状态键

#### 4.2.1 原始状态（Environment Raw State）

环境层核心输出包括：

| 字段 | 含义 |
|------|------|
| `inventory_raw` | 当前参考入住日的剩余库存 |
| `initial_inventory` | 初始库存 |
| `inventory_ratio` | 当前库存比例 |
| `future_inventory` | 未来91天库存曲线 |
| `day` | 当前仿真日 |
| `day_offset` | 当前轨道相对入住偏移 |

#### 4.2.2 补齐后的策略状态

`common.py::enrich_bucket_state()` 在原始状态基础上补齐：

- `season`
- `weekday`
- `inventory_level`
- `bucket_inv_ratio / near_inv_ratio / far_inv_ratio`
- `bucket_inv_bin / near_inv_bin / far_inv_bin`
- `inv_slope / inv_slope_bin`

其中库存离散采用5档，默认阈值为 `0.2 / 0.4 / 0.6 / 0.8`。

#### 4.2.3 Q-learning 的离散状态

Q-learning 仍使用标量状态索引：

$$
\mathcal{S}_{Q}=\{\text{inventory\_level}\}\times\{\text{season}\}\times\{\text{weekday}\}\times\{\text{stage\_id}\}
$$

其中：

- `inventory_level`: 5档
- `season`: 3档
- `weekday`: 2档
- `stage_id`: 8档

因此总状态数为：

$$5 \times 3 \times 2 \times 8 = 240$$

#### 4.2.4 CEM 的状态键

当前主线 `CEM` 不再强制使用单整数状态，而是使用更丰富的 tuple key：

$$
s_{\text{CEM}}=(\text{stage\_id}, \text{season}, \text{weekday}, \text{bucket\_inv\_bin}, \text{near\_inv\_bin}, \text{far\_inv\_bin}, \text{inv\_slope\_bin})
$$

该设计的目的不是压缩成一个固定大小的有限表，而是显式利用：

- 当前 bucket 的库存紧张度
- 近端与远端库存形状差异
- 决策阶段位置

因此，`CEM` 的有效状态数量由训练访问到的状态组合决定，不再简单等于 `18 x K` 或 `30 x K`。

### 4.3 动作空间设计（联合动作）

酒店Agent采用连续二维动作空间，联合输出两个价格：

| 动作维度 | 范围 | 说明 |
|----------|------|------|
| $P_{\text{online\_base}}$ | [50, 150] 元 | 给OTA的线上基础价格 |
| $P_{\text{offline}}$ | [50, 150] 元 | 线下直销价格 |

在 `cem` 模式下，动作采样来自二维高斯分布：

$$
\mathbf{a} = \begin{bmatrix}P_{\text{online\_base}} \\ P_{\text{offline}}\end{bmatrix}
\sim \mathcal{N}(\boldsymbol{\mu}_s,\boldsymbol{\Sigma}_s)
$$

其中 $\boldsymbol{\mu}_s \in \mathbb{R}^2$，$\boldsymbol{\Sigma}_s \in \mathbb{R}^{2\times2}$ 为状态 $s$ 的分布参数。

### 4.4 多元CEM更新机制（cem模式）

对每个状态 $s$，从经验中筛选精英动作集合 $\mathcal{E}$，计算：

$$
\hat{\boldsymbol{\mu}}=\text{mean}(\mathcal{E}), \quad
\hat{\boldsymbol{\Sigma}}=\text{cov}(\mathcal{E})
$$

平滑更新：

$$
\boldsymbol{\mu}_s \leftarrow (1-\alpha)\boldsymbol{\mu}_s+\alpha\hat{\boldsymbol{\mu}},\quad
\boldsymbol{\Sigma}_s \leftarrow (1-\alpha)\boldsymbol{\Sigma}_s+\alpha\hat{\boldsymbol{\Sigma}}
$$

并执行数值稳定处理：
- 协方差强制对称：$\Sigma\leftarrow(\Sigma+\Sigma^\top)/2$
- 对角线方差下限与正则化
- 特征值裁剪，确保半正定
- 仅对“本轮完成参数更新”的状态做协方差衰减

### 4.5 收益计算


$$R_{\text{hotel}} = \underbrace{B_{\text{online}} \times P_{\text{online\_base}} \times (1 - c)}_{\text{线上收益（扣佣金后）}} + \underbrace{B_{\text{offline}} \times P_{\text{offline}}}_{\text{线下收益}}$$

其中：
- $B_{\text{online}}$：线上渠道预订量
- $B_{\text{offline}}$：线下渠道预订量
- $c$：佣金率（默认 20%）

---

## 5. OTA策略设计（启发式外生环境）

### 5.1 设计理念

当前版本不再让OTA进行强化学习。OTA被建模为外生的反应式策略 `OTASubsidyHeuristic`，其输入为：
- 酒店线上基础价 $P_{\text{online\_base}}$
- 酒店线下价 $P_{\text{offline}}$
- 提前预订天数 `lead_time = t`

输出为补贴比例 $r_{\text{subsidy}} \in [0, r_{\max}]$。

### 5.2 时间敏感目标价差

OTA希望线上最终价格相对线下保持一定“竞争价差”，且该价差随提前期衰减：

$$
\Delta(t)=\Delta_{\max}\cdot e^{-\lambda t}
$$

其中默认 $\Delta_{\max}=15.0,\lambda=0.05$。

### 5.3 目标补贴率反推

线上最终价定义为：

$$
P_{\text{online\_final}} = P_{\text{online\_base}} \cdot (1-c\cdot r_{\text{subsidy}})
$$

由目标价差反推补贴率：

$$
r_{\text{target}}=\frac{P_{\text{online\_base}}-P_{\text{offline}}+\Delta(t)}{c\cdot P_{\text{online\_base}}}
$$

鲁棒性处理：若 $P_{\text{online\_base}}\le 0$ 或 $c\le 0$，直接返回 $0.0$。

### 5.4 约束与扰动

加入高斯噪声 $\epsilon\sim\mathcal{N}(0,\sigma^2)$，并裁剪到预算区间：

$$
r_{\text{actual}}=\text{clip}(r_{\text{target}}+\epsilon,0,r_{\max})
$$

默认 $r_{\max}=0.8,\sigma=0.05$。

### 5.5 价格计算链

酒店和OTA的动作共同决定最终的线上售价：

$$P_{\text{online\_final}} = P_{\text{online\_base}} - P_{\text{online\_base}} \times \text{commission\_rate} \times r_{\text{subsidy}}$$

即：最终线上价格 = 酒店基础价格 - OTA从佣金中拿出的补贴金额。

例如：酒店设定线上基础价格120元，佣金率30%，OTA补贴比例50%：
- 佣金收入 = 120 × 30% = 36元
- 补贴金额 = 36 × 50% = 18元
- 最终线上价格 = 120 - 18 = 102元

### 5.6 利润计算与统计

$$\Pi_{\text{OTA}} = \underbrace{B_{\text{online}} \times P_{\text{online\_base}} \times c}_{\text{佣金收入}} \times (1 - r_{\text{subsidy}})$$

等价于：

$$\Pi_{\text{OTA}} = \text{佣金收入} - \text{补贴支出}$$

其中 $\text{补贴支出} = \text{佣金收入} \times r_{\text{subsidy}}$。

注：OTA在当前版本不参与参数学习，但会持续记录 `total_profit / total_commission / total_subsidy_cost` 等统计指标。

---

## 6. Agent与环境的交互机制

### 6.1 交互总体流程

每天的交互涉及多个组件的协作：

```
┌──────────────┐     ①获取状态     ┌──────────────┐
│ Environment  │◄──────────────── │  GameTrainer  │
│ (hotel_env)  │                  │  (训练器)      │
│   ├─ state   │  ②状态传递       │               │
│   ├─ inventory│──────────────►  │ ┌───────────┐ │
│   └─ ABM     │                  │ │HotelAgent │ │
│              │                  │ │ → prices  │ │
│              │  ③价格传递       │ └───────────┘ │
│              │◄────────────────│               │
│              │                  │ ┌───────────┐ │
│              │  ④补贴传递       │ │OTA Heuristic││
│              │◄────────────────│ │ → subsidy  │ │
│              │                  │ └───────────┘ │
│              │                  │               │
│   ABM模拟     │  ⑤模拟结果      │               │
│   消费者到达   │──────────────►  │  ⑥计算收益    │
│   预订决策     │                 │  ⑦更新Hotel   │
│   库存扣减     │                 │               │
└──────────────┘                  └──────────────┘
```

### 6.2 91天滚动预订窗口

系统维护一个91天的滚动窗口（`booking_window_days = 91`），对应提前预订期0-90天：

```
Day t:    [t, t+1, t+2, ..., t+90]    ← 当前窗口
Day t+1:  [t+1, t+2, t+3, ..., t+91]  ← 窗口滚动
```

每天结束时：
1. 移除窗口最左端（当天已过期）
2. 在最右端添加新的一天（初始库存为满房）
3. 同步滚动价格窗口

每个窗口位置独立维护库存和价格，ABM消费者根据自身的提前预订期查找对应位置的价格。

### 6.3 决策桶（Decision Buckets）机制

91天的预订窗口被划分为多个决策桶（阶段），桶的主要作用是**定义“调价决策的阶段边界”**，而不是强制“桶内所有偏移永远使用同一个价格”。

在当前实现中，系统对每个入住日轨道（`day_offset`）单独维护挂牌价；当该轨道的 `day_offset` 进入某个桶的边界触发点（默认使用桶的右端点 `bucket_end`）时，才进行一次阶段切换调价。桶的粒度越细（越临近入住日），调价越频繁；桶越粗（远期），调价越稀疏。

默认分桶配置：`"0|1|2-3|4-6|7-13|14-29|30-59|60-90"`

| 桶编号 | 偏移范围 | 天数 | 语义 |
|--------|----------|------|------|
| 0 | 0 | 1天 | 当天入住 |
| 1 | 1 | 1天 | 明天入住 |
| 2 | 2-3 | 2天 | 后天-大后天 |
| 3 | 4-6 | 3天 | 近一周 |
| 4 | 7-13 | 7天 | 一至两周 |
| 5 | 14-29 | 16天 | 两周至一个月 |
| 6 | 30-59 | 30天 | 一至两个月 |
| 7 | 60-90 | 31天 | 两至三个月 |

**设计意图**：越近期的日期，定价决策越细致（单日级别）；越远期的日期，由于不确定性更大，使用更粗粒度的决策区间。

### 6.4 滚动窗口与状态同步

每天结束后，环境执行窗口滚动：

```
窗口滚动前: [Day_t, Day_{t+1}, ..., Day_{t+90}]
窗口滚动后: [Day_{t+1}, Day_{t+2}, ..., Day_{t+91}]
```

同步操作包括：
1. 从ABM模型回同步最新库存（ABM中已实时扣减）
2. 移除窗口最左端（今天已过期）
3. 在最右端添加新日期（初始库存=满房）
4. 价格窗口同步滚动
5. 决策桶的累计预订量同步滚动

### 6.5 详细交互时序

以一天（Day $t$）的交互为例：

**Step 1: 触发更新（按入住日轨道，在桶边界更新挂牌价）**

系统对窗口内每个入住日轨道（`day_offset`）维护独立的挂牌价与补贴率：
- `P_online_base[off]`：酒店线上基础价（给OTA计佣金的基础）
- `P_offline[off]`：酒店线下价
- `r_subsidy[off]`：OTA补贴比例

窗口 `0..90` 被 `decision_buckets` 切分为 K 个阶段（桶索引为 `stage_id`）。系统只在每个桶的右端点（`bucket_end`）触发该入住日轨道的“阶段切换调价”：

1. 对每个触发偏移 `off ∈ {bucket_end}`：
   - 先对该轨道上一阶段累计预订量做阶段结算（见 Step 3），并用结算奖励更新上一阶段决策
   - 再读取状态并做新阶段决策：
     ```python
     state = env._get_state_for_day_offset(off)
     state["stage_id"] = bucket_of_offset[off]  # 桶索引
     P_online_base[off], P_offline[off] = hotel_agent.select_action(state)
     r_subsidy[off] = ota_agent.get_subsidy(P_online_base[off], P_offline[off], lead_time=off)
     ```

**Step 2: 环境模拟阶段（整窗口执行）**

用当前所有轨道的挂牌价构造完整的91天价格窗口，并执行环境一步：

1. 最终线上价：
   $$P_{\text{online,final}}[off] = P_{\text{online,base}}[off] - P_{\text{online,base}}[off]\cdot c \cdot r_{\text{subsidy}}[off]$$

2. 传入环境：
   ```python
   actions_window = [[P_online_final[i], P_offline[i]] for i in range(91)]
   _, _, done, info = env.step(actions_window)
   ```

环境内部：
1. 将价格窗口同步到 ABM（按 `days_ahead` 取价）
2. 将未来库存同步到 ABM（按入住日 `target_date` 扣减）
3. ABM 执行当日模拟（消费者生成→决策→预订→库存扣减）
4. 返回 `bookings_by_day_offset`（按入住日轨道/提前期统计的预订量与收入）

**Step 3: 奖励计算与阶段结算更新（credit assignment）**

1. 从 `info['bookings_by_day_offset']` 读取每个 `off` 的预订量 `B_online[off]`、`B_offline[off]`
2. 将其累加到该轨道的“阶段累计预订量”中（用于下一次触发点结算）
3. 当 `off` 命中触发点时，使用该轨道上一阶段累计预订量计算酒店收益与OTA利润，并仅更新酒店Agent参数（OTA为外生规则，不更新）
---

## 7. 奖励函数设计

### 7.1 基本收益计算

**酒店原始收益**：

$$R_{\text{hotel}}^{\text{raw}} = B_{\text{online}} \times P_{\text{online\_base}} \times (1 - c) + B_{\text{offline}} \times P_{\text{offline}}$$

**OTA原始利润**：

$$\Pi_{\text{OTA}}^{\text{raw}} = B_{\text{online}} \times P_{\text{online\_base}} \times c \times (1 - r_{\text{subsidy}})$$

**系统总利润**：

$$\Pi_{\text{system}} = R_{\text{hotel}}^{\text{raw}} + \Pi_{\text{OTA}}^{\text{raw}}$$

### 7.2 酒店训练奖励（当前生效）

当前主线中 `reward_hotel_ratio = 1`，因此酒店的基础训练奖励等于酒店原始收益：

$$
\text{BaseReward}_{\text{hotel}} = R_{\text{hotel}}^{\text{raw}}
$$

在此基础上，再叠加轻量的机会成本惩罚（reward shaping）：

$$
\text{Reward}_{\text{hotel}}=\max \left(0,\ \text{BaseReward}_{\text{hotel}}-\text{Penalty}_{\text{price}}-\text{Penalty}_{\text{sellthrough}}\right)
$$

其中惩罚项由 `src/utils/common.py::compute_reward_shaping()` 计算，核心包含：

- `pressure`：库存压力，由 `bucket_inv_ratio` 与 `near/far` 库存差共同决定
- `low_price_signal`：价格越接近下界，该信号越高
- `sellthrough_excess`：卖得超过目标节奏 `target_sellthrough` 的部分

具体形式为：

$$
\text{Penalty}_{\text{price}} \propto w_p \cdot \text{pressure}^{1.4}\cdot \text{low\_price\_signal}^{1.25}
$$

$$
\text{Penalty}_{\text{sellthrough}} \propto w_s \cdot \text{pressure}^{1.2}\cdot \text{sellthrough\_excess}^{1.15}
$$

其设计意图是：在库存稀缺时，对“低价快卖”施加明确的机会成本，从而更容易改变 `CEM` 的 elite 排序。

OTA 不接收学习奖励，仅按启发式规则输出补贴并统计利润。

### 7.3 奖励的时间聚合

奖励不是逐天给予酒店Agent的，而是按**决策桶的触发时刻**聚合：

```
桶 k 的触发日到达时:
  reward_k = 该桶内所有日期偏移的累计预订产生的总奖励
```

这意味着酒店Agent看到的是一段时间内（桶跨度）策略执行的累计效果。当前日志中会同时记录：

- `TrainBase`：shaping 前的基础训练奖励
- `TrainShaped`：实际送入 `CEM.update()` 的训练奖励
- `ShapePenalty`：平均惩罚比例

---

## 8. 伪代码（当前实现）

```
输入: 历史数据, 训练轮数 E, 预订窗口 W=91, 决策桶配置
输出: 训练后的酒店Agent（OTA为启发式规则）

1. 初始化环境 env, 酒店Agent, OTA启发式策略
2. 解析决策桶 → K 个阶段

3. FOR episode = 1 to E:
   a. env.reset()
   b. 对每个桶 k ∈ {0,...,K-1}:
      - 获取桶 k 代表位置的状态 s_k
      - 酒店选择价格: (P_on[k], P_off[k]) = hotel.select_action(s_k)
      - OTA计算补贴: r_sub[k] = ota.get_subsidy(P_on[k], P_off[k], lead_time)
      - 初始化桶 k 的累计预订量 = 0

   c. FOR day = 0 to 364:
      (i)  对每个触发偏移 off:
           - 若该偏移的累计预订 > 0:
             · 计算酒店收益和OTA利润
             · 构建酒店奖励信号
             · 更新酒店CEM参数
             · 重置累计预订 = 0
           - 重新决策该偏移的价格和补贴

      (ii) 计算最终线上价格窗口
      (iii) 传入环境: env.step(actions_window)
      (iv) 从返回的 bookings_by_day_offset 中累计各偏移的预订量

      (v)  按 update_frequency 定期调用 hotel.end_episode()
           → 触发酒店CEM参数更新与探索衰减

      (vi) 滚动所有窗口数组

   d. 处理剩余未更新的桶（episode结束时的最终更新）
   e. 调用 hotel.end_episode() 完成episode级参数更新
   f. 记录episode统计信息

4. 保存酒店模型参数与OTA策略配置快照
```

---

## 9. 数据来源与预处理

### 9.1 数据集

系统使用 **Hotel Booking Demand** 数据集（`hotel_bookings.csv`），并仅使用其中 **City Hotel**（城市酒店）的子集。

### 9.2 从历史数据中提取的参数

| 提取内容 | 使用的字段 | 处理方法 |
|----------|-----------|---------|
| 月度到达率 $\lambda_m$ | `arrival_date_month` | 按月分组计数后除以30 |
| 提前期经验分布 | `lead_time` | 统计0-90天各天数的频率，构建离散概率分布 |
| WTP分布参数 | `adr` | 未取消订单的ADR的均值和标准差 |

### 9.3 数据过滤

- 仅使用 `hotel == 'City Hotel'` 的记录
- WTP拟合时排除已取消订单（`is_canceled == 1`）
- 排除ADR异常值（$\leq 0$ 或 $\geq 500$）
- 提前期裁剪到 [0, 90] 天

---

## 10. 实验配置与超参数

### 10.1 环境参数

| 参数 | 值 | 说明 |
|------|----|----|
| `initial_inventory` | 70 | 酒店总客房数 |
| `booking_window_days` | 91 | 预订窗口长度（含当天） |
| `episode_days` | 730 | 每个episode模拟天数 |
| `cost_per_room` | 20 | 每间客房成本 |

### 10.2 博弈系统参数（当前主线）

| 参数                    | 值            | 说明        |
| --------------------- | ------------ | --------- |
| `commission_rate`     | 0.20         | OTA佣金率    |
| `subsidy_ratio_max`   | 0.8          | OTA最高补贴比例 |
| `ota_delta_max`       | 15.0         | OTA目标价差上限 |
| `ota_decay_lambda`    | 0.05         | 提前期衰减系数 |
| `ota_noise_std`       | 0.05         | 补贴噪声标准差 |
| `ota_seed`            | 42           | OTA随机种子 |
| `online_price_range`  | [50, 150]    | 线上基础价格范围  |
| `offline_price_range` | [50, 150]    | 线下价格范围    |
| `training_mode`       | simultaneous | 默认同步训练    |

### 10.3 ABM消费者参数

| 参数                               | 值          | 说明          |
| -------------------------------- | ---------- | ----------- |
| `urgency_weight` $\gamma$        | 20         | 紧迫性权重       |
| `booking_threshold` $\theta$     | -15        | 预订效用阈值      |
| `customer_type_ratio`            | (0.7, 0.3) | (online_only, omnichannel) |
| `online_discount_ratio` $\delta$ | 0.95       | 线上渠道WTP折扣系数 |
| `noise_std`                      | 12.0       | 效用噪声标准差     |

### 10.4 酒店CEM参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cem_algorithm` | `cem` | `cem`为多元高斯版，`cem_nn`为神经网络版 |
| `cem_n_samples` | 400 | 采样数量 |
| `cem_elite_frac` | 0.3 | 精英比例 |
| `initial_std` | 50.0 | 初始探索标准差 |
| `min_std` | 3.0 | 最小探索标准差 |
| `std_decay` | 0.999 | 探索衰减系数 |

### 10.5 训练参数

| 参数                   | 默认值                                         | 说明             |
| -------------------- | ------------------------------------------- | -------------- |
| `episodes`           | 250（配置默认）/ 50（CLI默认）                      | 训练总轮数          |
| `update_frequency`   | 30                                          | CEM分布更新频率（每N天） |
| `decision_buckets`   | `0\|1\|2-3\|4-6\|7-13\|14-29\|30-59\|60-90` | 提前期分桶配置        |
| `reward_hotel_ratio` | 1.0                                         | 酒店个体收益权重       |
| `reward_shape_price_weight` | 0.30                                | 低价出清惩罚强度      |
| `reward_shape_sellthrough_weight` | 0.22                          | 过快售出惩罚强度      |
| `reward_shape_target_sellthrough` | 0.25                          | 节奏惩罚阈值         |

---

## 11. Cross-Entropy Method算法实现（酒店）

### 11.1 算法概述

Cross-Entropy Method（CEM）是一种基于采样的黑箱优化方法，不需要计算梯度，通过迭代地采样-筛选-更新来搜索最优策略。其核心思想是：

1. 维护每个状态的动作分布（高斯分布）
2. 从分布中采样多个候选动作
3. 根据收益选出"精英"样本
4. 用精英样本的统计量更新分布参数

CEM特别适合本系统的原因：
- **无需梯度**：环境奖励通过ABM模拟获得，不可微分
- **高稳定性**：不存在Q值高估等问题，训练过程稳定
- **适合随机环境**：ABM环境具有固有随机性（泊松到达、随机WTP等），CEM天然适应

### 11.2 数学形式化（多元高斯）

对每个状态 $s$，维护二维高斯分布参数 $(\boldsymbol{\mu}_s, \boldsymbol{\Sigma}_s)$：

$$\mathbf{a}\sim\mathcal{N}(\boldsymbol{\mu}_s,\boldsymbol{\Sigma}_s),\ \mathbf{a}\in\mathbb{R}^2$$

#### 11.2.1 经验收集

每次在状态 $s$ 下执行二维动作 $\mathbf{a}$ 并获得奖励 $r$ 后，将 $(s,\mathbf{a},r)$ 存入状态回放缓冲区（`deque(maxlen=memory_size)`）。

#### 11.2.2 分布更新

在每个episode结束时（或按固定频率），对所有被访问过的状态执行分布更新：

1. **取最近 $N$ 个经验**：$\{(a_1, r_1), (a_2, r_2), \ldots, (a_N, r_N)\}$，其中 $N = \min(\text{n\_samples}, |\text{memory}[s]|)$

2. **选择精英样本**：按奖励排序，取前 $k$ 个最优动作：
   $$\mathcal{E} = \text{top-}k\{a_j\}_{j=1}^{N} \text{ by } r_j, \quad k = \lfloor N \times \text{elite\_frac} \rfloor$$

3. **计算精英统计量**：
   $$\hat{\boldsymbol{\mu}}=\text{mean}(\mathcal{E}),\quad \hat{\boldsymbol{\Sigma}}=\text{cov}(\mathcal{E})$$

4. **平滑更新分布参数**（学习率 $\alpha$）：
   $$\boldsymbol{\mu}_s\leftarrow(1-\alpha)\boldsymbol{\mu}_s+\alpha\hat{\boldsymbol{\mu}}$$
   $$\boldsymbol{\Sigma}_s\leftarrow(1-\alpha)\boldsymbol{\Sigma}_s+\alpha\hat{\boldsymbol{\Sigma}}$$

5. **稳定性处理**：
   - 对称化：$\Sigma\leftarrow(\Sigma+\Sigma^\top)/2$
   - 对角线方差下限与正则项
   - 特征值裁剪，保证半正定

6. **探索衰减**：仅对本轮完成更新的状态做协方差缩放

### 11.3 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_samples` | 100 | 每次评估时使用的最近经验数量 |
| `elite_frac` | 0.3 | 精英样本比例，取奖励最高的前30% |
| `initial_std` | 20.0 | 初始标准差（二维动作） |
| `min_std` | 5.0 | 方差下限对应标准差 |
| `std_decay` | 0.999 | 协方差衰减系数 |
| `memory_size` | 100 | 每个状态的经验回放缓冲区大小 |
| 更新率 $\alpha$ | 0.3 | 分布参数平滑更新步长 |

### 11.4 探索与利用

酒店CEM的探索由协方差矩阵控制：
- **初期**：方差较大，联合探索线上/线下价格组合
- **中期**：精英更新后逐步收敛
- **后期**：协方差接近下限，策略趋于稳定

### 11.5 OTA算法说明

OTA在当前版本不使用CEM，采用启发式外生规则：
- 输入：`(P_online_base, P_offline, lead_time)`
- 输出：`r_subsidy = clip(r_target + noise, 0, r_max)`
- 作用：提供稳定、可解释且可控的补贴响应机制

---

## 12. 关键设计决策总结

### 12.1 为什么选择CEM而非DQN/PPO等算法？

- **稳定性**：CEM不存在Q值高估、策略振荡等深度RL常见问题
- **无需梯度**：ABM环境是黑箱模拟器，奖励信号不可微分
- **适合随机环境**：CEM通过精英采样自然平滑了环境随机性
- **简洁高效**：表格版CEM实现简单，计算开销小
- **可解释性**：CEM的分布参数（均值和标准差）直接对应Agent在各状态下的定价策略和不确定性

### 12.2 为什么改为“酒店多元CEM + OTA启发式”？

核心原因：
- 酒店两渠道价格存在显著联动，多元CEM可通过协方差直接学习交叉关系
- OTA改为外生规则后，系统训练稳定性更高、解释性更强
- 便于进行政策/参数敏感性实验（`delta_max`、`decay_lambda`、`noise_std`）

### 12.3 决策桶的设计意义

决策桶仍然是当前系统的关键设计，但其作用已经从“简单压缩固定状态表”扩展为：
- 为滚动窗口中的不同 lead time 轨道提供稳定的阶段边界
- 支持 `Q-learning` 的 `240` 个离散状态
- 为 `CEM` 的 richer tuple state 提供 `stage_id / bucket_start / bucket_end` 上下文

因此，分桶的意义不仅是减少状态数量，更重要的是让价格更新频率与 lead time 结构对齐。

### 12.4 当前奖励设计的意义

当前主线不再强调“酒店/系统混合奖励”，而是采用：

- **主目标**：酒店收益最大化（`reward_hotel_ratio = 1`）
- **辅助约束**：在库存稀缺时，对低价和过快售出施加轻量机会成本惩罚

其优点是：
- 与最终评估指标一致
- 保持业务可解释性
- 更适合 `CEM` 这种基于样本排序的优化方法

---

> **文档版本**: v2.1
>
> **最后更新**: 2026年5月20日
>
> **适用代码版本**: 酒店多元CEM/CEM-NN + OTA启发式补贴 + richer state + reward shaping 版本
