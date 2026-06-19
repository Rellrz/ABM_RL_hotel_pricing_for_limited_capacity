#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABM客户行为模型 - 基于Mesa框架

主要功能：
1. 客户生成模块：基于泊松分布生成每日潜在客户
2. 客户决策模块：基于效用函数的预订决策
3. 客户取消模块：动态持有效用评估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from mesa import Agent, Model
from mesa.datacollection import DataCollector
import warnings
from configs.config import ABM_CONFIG
warnings.filterwarnings('ignore')


@dataclass
class CustomerProfile:
    """客户特征配置文件"""
    lead_time: int              # 提前预订期（天）
    target_date: int            # 目标入住日期（仿真日期）
    wtp: float                  # 最高支付意愿（Willingness To Pay）
    price_sensitivity: float    # 价格敏感度系数 β
    customer_type: str          # 客户分群：'online_only' 或 'omnichannel'


@dataclass
class BookingRecord:
    """预订记录"""
    customer_id: int
    booking_date: int           # 预订日期
    target_date: int            # 入住日期
    paid_price: float           # 成交价格
    wtp: float                  # 支付意愿
    price_sensitivity: float    # 价格敏感度
    customer_type: str          # 最终成交渠道：'online' 或 'offline'
    is_canceled: bool = False   # 是否已取消


class CustomerAgent(Agent):
    """
    客户智能体
    
    每个客户代表一个潜在的酒店预订者，具有独特的特征和决策逻辑。
    """
    
    def __init__(self, unique_id: int, model: 'HotelABMModel', profile: CustomerProfile):
        """
        初始化客户智能体
        
        Args:
            unique_id: 唯一标识符
            model: ABM模型实例
            profile: 客户特征配置
        """
        super().__init__(unique_id, model)
        self.profile = profile
        self.has_booked = False
        self.booking_record: Optional[BookingRecord] = None
        self.last_decision_details: Optional[Dict[str, object]] = None

    def _anchor_value(self, price_gap: float) -> float:
        lambda_plus = float(max(0.0, getattr(self.model.params, 'anchor_lambda_plus', 1.0)))
        lambda_minus = float(max(0.0, getattr(self.model.params, 'anchor_lambda_minus', 2.0)))
        if price_gap >= 0.0:
            return lambda_plus * float(price_gap)
        return lambda_minus * float(price_gap)

    def _compute_anchor_utility(self, channel: str, price: float, reference_prices: Dict[str, float]) -> float:
        if not bool(reference_prices.get('enabled', False)):
            return 0.0

        del channel
        price = float(price)
        anchor_eta = float(max(0.0, getattr(self.model.params, 'anchor_eta', 0.30)))
        customer_type = self.profile.customer_type
        if customer_type == 'online_only':
            irp_single = float(reference_prices.get('single', price))
            return float(anchor_eta * self._anchor_value(irp_single - price))

        irp_joint = float(reference_prices.get('joint', price))
        return float(anchor_eta * self._anchor_value(irp_joint - price))
        
    def evaluate_booking_utility(
        self,
        price: float,
        current_day: int,
        reference_prices: Dict[str, float],
        discount_ratio: float = 1.0,
        channel: str = 'online',
    ) -> float:
        """
        评估预订效用
        
        基于效用函数计算客户的预订意愿：
        U_score = (WTP - P) * β + γ/(L+1) + ε
        
        Args:
            price: 酒店当前报价
            current_day: 当前仿真日期
            
        Returns:
            效用得分
        """
        # 经济盈余效用：β * (WTP - P)
        economic_surplus = self.profile.price_sensitivity * ((self.profile.wtp * discount_ratio) - price)
        
        anchor_utility = self._compute_anchor_utility(
            channel=channel,
            price=float(price),
            reference_prices=reference_prices,
        )

        # 紧迫性效用：γ/(L+1)
        # 提前期越短，紧迫性越高
        gamma = self.model.params.urgency_weight
        urgency_utility = gamma / (self.profile.lead_time + 1)

        
        # 总效用 + 行为噪声（可开关）
        utility = (
            economic_surplus
            + anchor_utility
            + urgency_utility
            + self.model.sample_utility_noise(current_day)
        )
        
        return utility
    
    def make_booking_decision(
        self,
        online_price: float,
        offline_price: float,
        current_day: int,
        anchor_reference_prices: Dict[str, Dict[str, float]],
    ) -> bool:
        """
        做出预订决策
        
        Args:
            online_price: 线上渠道报价
            current_day: 当前日期
            
        Returns:
            是否预订
        """
        if self.has_booked:
            return False
        
        # 计算效用
        if self.profile.customer_type == 'online_only':
            online_utility = self.evaluate_booking_utility(
                online_price,
                current_day,
                reference_prices=anchor_reference_prices['online_only'],
                discount_ratio=self.model.params.online_discount_ratio,
                channel='online',
            )
            offline_utility = None
        else:
            online_utility = self.evaluate_booking_utility(
                online_price,
                current_day,
                reference_prices=anchor_reference_prices['omnichannel'],
                discount_ratio=self.model.params.online_discount_ratio,
                channel='online',
            )
            offline_utility = self.evaluate_booking_utility(
                offline_price,
                current_day,
                reference_prices=anchor_reference_prices['omnichannel'],
                channel='offline',
            )
        
        if offline_utility is None:
            utility = online_utility
            price = online_price
            chosen_channel = 'online'
        else:
            if online_utility > offline_utility:
                utility = online_utility
                price = online_price
                chosen_channel = 'online'
            else:
                utility = offline_utility
                price = offline_price
                chosen_channel = 'offline'

        threshold = self.model.params.booking_threshold
        decision_pass = bool(utility > threshold)
        if bool(getattr(self.model, "trace_customer_utility", False)):
            self.last_decision_details = {
                'online_utility': float(online_utility),
                'offline_utility': float(np.nan if offline_utility is None else offline_utility),
                'chosen_channel': str(chosen_channel),
                'chosen_price': float(price),
                'chosen_utility': float(utility),
                'booking_threshold': float(threshold),
                'passed_utility_threshold': decision_pass,
            }
        else:
            self.last_decision_details = None
        # 做出决策
        if decision_pass:
            self.has_booked = True
            self.booking_record = BookingRecord(
                customer_id=self.unique_id,
                booking_date=current_day,
                target_date=self.profile.target_date,
                paid_price=price,
                wtp=self.profile.wtp,
                price_sensitivity=self.profile.price_sensitivity,
                customer_type=chosen_channel
            )
            return True
        
        return False
    
    def evaluate_holding_utility(self, current_price: float, days_until_checkin: int) -> float:
        """
        评估持有订单的效用（用于取消决策）
        
        U_hold = (WTP - P_paid) - β * max(0, P_paid - P_curr) + γ/(d+1) + ξ
        
        Args:
            current_price: 当前酒店报价
            days_until_checkin: 距离入住的剩余天数
            
        Returns:
            持有效用得分
        """
        if not self.has_booked or self.booking_record is None:
            return 0.0
        
        # 原始满意度：(WTP - P_paid)
        satisfaction = self.profile.wtp - self.booking_record.paid_price
        
        # 价格后悔：β * max(0, P_paid - P_curr)
        regret_coef = self.model.params['regret_coefficient']
        price_regret = regret_coef * max(0, self.booking_record.paid_price - current_price)
        
        # 临近承诺效应：γ/(d+1)
        commitment_weight = self.model.params['commitment_weight']
        commitment_utility = commitment_weight / (days_until_checkin + 1)
        
        # 每日随机冲击
        shock_std = self.model.params['shock_std']
        daily_shock = np.random.normal(0, shock_std)
        
        # 总持有效用
        holding_utility = satisfaction - price_regret + commitment_utility + daily_shock
        
        return holding_utility
    
    def evaluate_cancellation(self, current_price: float, current_day: int) -> bool:
        """
        评估是否取消订单
        
        Args:
            current_price: 当前酒店报价
            current_day: 当前日期
            
        Returns:
            是否取消
        """
        if not self.has_booked or self.booking_record is None or self.booking_record.is_canceled:
            return False
        
        # 计算距离入住的天数
        days_until_checkin = self.booking_record.target_date - current_day
        
        # 如果已经是入住日或过期，不取消
        if days_until_checkin <= 0:
            return False
        
        # 计算持有效用
        holding_utility = self.evaluate_holding_utility(current_price, days_until_checkin)
        
        # 如果持有效用为负，取消订单
        if holding_utility < 0:
            self.booking_record.is_canceled = True
            return True
        
        return False
    
    def step(self):
        """
        智能体的每步行为（由Mesa框架调用）
        """
        pass


class HotelABMModel(Model):
    """
    酒店ABM模型
    
    模拟酒店预订环境，包括客户生成、决策和取消行为。
    """
    
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 random_seed: Optional[int] = None,
                 booking_window_days: int = 5):
        """
        初始化ABM模型
        
        Args:
            historical_data: 历史预订数据（hotel_bookings.csv）
            params: 模型参数字典
            random_seed: 随机种子
        """
        super().__init__()
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            self.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        
        # 存储历史数据
        self.historical_data = historical_data
        
        # 转换为字典格式（保持向后兼容）
        self.params = ABM_CONFIG
        
        # 初始化调度器，随机策略
        # 当前仿真日期
        self.current_day = 0
        
        # ✅ 每日可用库存字典（由HotelEnvironment同步）
        from collections import defaultdict
        self.daily_available_rooms = defaultdict(lambda: 226)
        
        # ✅ 价格窗口（由HotelEnvironment同步）
        self.booking_window_days = int(booking_window_days)
        self.price_window_online = [100.0] * self.booking_window_days
        self.price_window_online_base = [100.0] * self.booking_window_days
        self.price_window_offline = [120.0] * self.booking_window_days
        self.commission_rate = 0.0

        # 扰动状态（OU/AR1）
        self.demand_ou_state = 0.0
        self.wtp_ou_state = 0.0
        self.channel_ou_state = 0.0
        
        # 活跃预订记录（未取消且未入住）
        self.active_bookings: List[BookingRecord] = []
        
        # 历史记录
        self.booking_history: List[BookingRecord] = []
        self.daily_stats: List[Dict] = []
        self.trace_customer_utility = False
        self.customer_utility_trace: List[Dict] = []
        self.total_customers_generated = 0
        self.total_bookings_count = 0
        self.total_cancellations_count = 0
        self._calendar_feature_cache: Dict[int, Tuple[int, int, int]] = {}
        
        # 数据收集器
        self.datacollector = self._build_datacollector()

    def _build_datacollector(self) -> DataCollector:
        return DataCollector(
            model_reporters={
                "total_customers": lambda m: m.total_customers_generated,
                "total_bookings": lambda m: m.total_bookings_count,
                "total_cancellations": lambda m: m.total_cancellations_count,
                "active_bookings": lambda m: len(m.active_bookings),
            }
        )

    def _calendar_features(self, current_day: int) -> Tuple[int, int, int]:
        cached = self._calendar_feature_cache.get(current_day)
        if cached is not None:
            return cached
        month = (current_day // 30) % 12 + 1
        if month in [11, 12, 1, 2]:
            season = 0
        elif month in [6, 7, 8]:
            season = 2
        else:
            season = 1
        is_weekend = 1 if (current_day % 7) in [5, 6] else 0
        features = (month, season, is_weekend)
        self._calendar_feature_cache[current_day] = features
        return features

    def _ou_step(self, x: float, theta: float, sigma: float) -> float:
        return (1.0 - float(theta)) * float(x) + float(sigma) * float(self.rng.normal())

    def _apply_demand_perturbation(self, lambda_base: float) -> float:
        if not bool(self.params.enable_perturbation):
            return max(1e-6, float(lambda_base))

        self.demand_ou_state = self._ou_step(
            self.demand_ou_state,
            self.params.demand_ou_theta,
            self.params.demand_ou_sigma,
        )
        jump = 0.0
        if float(self.rng.random()) < float(self.params.demand_jump_prob):
            jump = float(self.rng.normal(self.params.demand_jump_mean, self.params.demand_jump_std))
        multiplier = float(np.exp(self.demand_ou_state + jump))
        multiplier = float(np.clip(multiplier, self.params.lambda_multiplier_min, self.params.lambda_multiplier_max))
        return max(1e-6, float(lambda_base) * multiplier)

    def _apply_wtp_perturbation(self, wtp_mean: float, wtp_std: float) -> Tuple[float, float]:
        if not bool(self.params.enable_perturbation):
            return float(wtp_mean), float(max(1e-6, wtp_std))

        self.wtp_ou_state = self._ou_step(
            self.wtp_ou_state,
            self.params.wtp_ou_theta,
            self.params.wtp_ou_sigma,
        )
        mean_mult = float(np.exp(self.wtp_ou_state))
        mean_mult = float(np.clip(mean_mult, self.params.wtp_multiplier_min, self.params.wtp_multiplier_max))
        std_mult = float(np.clip(mean_mult, self.params.wtp_std_multiplier_min, self.params.wtp_std_multiplier_max))
        return float(wtp_mean) * mean_mult, max(1e-6, float(wtp_std) * std_mult)

    def _apply_channel_online_only_prob(self, base_online_only_prob: float) -> float:
        p = float(np.clip(base_online_only_prob, 0.0, 1.0))
        if not bool(self.params.enable_perturbation):
            return p

        self.channel_ou_state = self._ou_step(
            self.channel_ou_state,
            self.params.channel_pref_ou_theta,
            self.params.channel_pref_ou_sigma,
        )
        p = p + self.channel_ou_state
        p = float(np.clip(p, self.params.channel_online_only_prob_min, self.params.channel_online_only_prob_max))
        return p

    def sample_utility_noise(self, current_day: int) -> float:
        del current_day
        if not bool(self.params.enable_perturbation):
            return 0.0

        noise_type = str(self.params.utility_noise_type).lower()
        if noise_type == 'gumbel':
            beta = max(1e-6, float(self.params.utility_gumbel_beta))
            return float(self.rng.gumbel(loc=0.0, scale=beta))
        if noise_type == 'normal':
            std = max(0.0, float(self.params.utility_normal_std))
            return float(self.rng.normal(loc=0.0, scale=std))
        return 0.0

    def _get_price_calendar(self, channel: str) -> np.ndarray:
        if channel == 'online':
            prices = self.price_window_online
        else:
            prices = self.price_window_offline
        arr = np.asarray(prices, dtype=float)
        if arr.size == 0:
            return np.asarray([0.0], dtype=float)
        return arr

    def _compute_internal_reference_price(self, prices: np.ndarray) -> float:
        q_low = float(np.clip(getattr(self.params, 'anchor_quantile_low', 0.10), 0.0, 1.0))
        q_high = float(np.clip(getattr(self.params, 'anchor_quantile_high', 0.90), 0.0, 1.0))
        if q_low > q_high:
            q_low, q_high = q_high, q_low

        w_low = float(max(0.0, getattr(self.params, 'anchor_weight_low', 0.50)))
        w_mean = float(max(0.0, getattr(self.params, 'anchor_weight_mean', 0.35)))
        w_high = float(max(0.0, getattr(self.params, 'anchor_weight_high', 0.15)))
        w_sum = w_low + w_mean + w_high
        if w_sum <= 0:
            w_low, w_mean, w_high = 0.50, 0.35, 0.15
            w_sum = 1.0

        p_low = float(np.quantile(prices, q_low))
        p_mean = float(np.mean(prices))
        p_high = float(np.quantile(prices, q_high))
        return float((w_low * p_low + w_mean * p_mean + w_high * p_high) / w_sum)

    def _record_customer_utility_trace(
        self,
        *,
        customer: CustomerAgent,
        current_day: int,
        day_offset: int,
        online_price: float,
        online_price_base: float,
        offline_price: float,
        inventory_before: int,
        inventory_after: int,
        final_status: str,
    ) -> None:
        if not bool(getattr(self, "trace_customer_utility", False)):
            return

        decision = customer.last_decision_details or {}
        self.customer_utility_trace.append(
            {
                'current_day': int(current_day),
                'customer_id': int(customer.unique_id),
                'lead_time': int(customer.profile.lead_time),
                'target_date': int(customer.profile.target_date),
                'day_offset': int(day_offset),
                'customer_type': str(customer.profile.customer_type),
                'wtp': float(customer.profile.wtp),
                'price_sensitivity': float(customer.profile.price_sensitivity),
                'online_price': float(online_price),
                'online_price_base': float(online_price_base),
                'offline_price': float(offline_price),
                'online_utility': float(decision.get('online_utility', np.nan)),
                'offline_utility': float(decision.get('offline_utility', np.nan)),
                'chosen_channel': str(decision.get('chosen_channel', 'none')),
                'chosen_price': float(decision.get('chosen_price', np.nan)),
                'chosen_utility': float(decision.get('chosen_utility', np.nan)),
                'booking_threshold': float(decision.get('booking_threshold', np.nan)),
                'passed_utility_threshold': bool(decision.get('passed_utility_threshold', False)),
                'inventory_before': int(inventory_before),
                'inventory_after': int(inventory_after),
                'booked': bool(final_status.startswith('booked_')),
                'final_status': str(final_status),
            }
        )


    def build_daily_anchor_reference_prices(self) -> Dict[str, Dict[str, float]]:
        if not bool(getattr(self.params, 'anchor_enabled', False)):
            return {
                'online_only': {'enabled': False},
                'omnichannel': {'enabled': False},
            }

        prices_on = self._get_price_calendar('online')
        irp_on = self._compute_internal_reference_price(prices_on)
        prices_off = self._get_price_calendar('offline')
        irp_off = self._compute_internal_reference_price(prices_off)
        theta = float(np.clip(getattr(self.params, 'anchor_joint_theta', 0.50), 0.0, 1.0))
        irp_joint = theta * irp_on + (1.0 - theta) * irp_off
        return {
            'online_only': {
                'enabled': True,
                'single': float(irp_on),
            },
            'omnichannel': {
                'enabled': True,
                'online': float(irp_on),
                'offline': float(irp_off),
                'joint': float(irp_joint),
            },
        }
    
    def generate_daily_customers(self, current_day: int) -> List[CustomerAgent]:
        """
        生成当日的潜在客户
        
        Args:
            current_day: 当前仿真日期
            
        Returns:
            客户智能体列表
        """
        # 确定当前月份与日类型（工作日/节假日，当前节假日用周末代理）
        month, _, is_weekend = self._calendar_features(current_day)

        # 优先使用月×日类型到达率；缺失时回退到月度到达率
        monthly_daytype_rates = getattr(self.params, 'arrival_rate_by_month_daytype', {}) or {}
        month_rates = monthly_daytype_rates.get(month, {}) if isinstance(monthly_daytype_rates, dict) else {}
        lambda_base = month_rates.get(is_weekend, self.params.monthly_arrival_rates.get(month, 100.0))
        lambda_eff = self._apply_demand_perturbation(lambda_base)
        
        # 从泊松分布采样当日客户数量
        num_customers = int(self.rng.poisson(lambda_eff))
        self.total_customers_generated += num_customers
        
        # 生成客户
        customers = []
        for _ in range(num_customers):
            # 生成唯一ID
            customer_id = self.next_id()
            
            # 生成客户特征
            profile = self._generate_customer_profile(current_day)
            
            # 创建客户智能体
            customer = CustomerAgent(customer_id, self, profile)
            customers.append(customer)
        
        return customers
    
    def _sample_lead_time(self, current_day: int) -> int:
        lead_time_params = self.params.lead_time_params
        dist_type = lead_time_params.get('type', 'exponential')

        if dist_type == 'empirical':
            _, season, is_weekend = self._calendar_features(current_day)

            conditional = lead_time_params.get('conditional_probabilities')
            if isinstance(conditional, dict):
                season_map = conditional.get(season)
                if isinstance(season_map, dict):
                    probs = season_map.get(is_weekend)
                    support = lead_time_params.get('support')
                    if support is not None and probs is not None and len(support) == len(probs) and len(support) > 0:
                        lead_time = int(self.rng.choice(support, p=probs))
                        return max(0, min(lead_time, self.booking_window_days - 1))

            support = lead_time_params.get('support')
            probabilities = lead_time_params.get('probabilities')
            if support is not None and probabilities is not None and len(support) == len(probabilities) and len(support) > 0:
                lead_time = int(self.rng.choice(support, p=probabilities))
                return max(0, min(lead_time, self.booking_window_days - 1))

        mean = float(lead_time_params.get('mean', 104.0))
        lead_time = int(self.rng.exponential(mean))
        return max(0, min(lead_time, self.booking_window_days - 1))

    def _generate_customer_profile(self, current_day: int) -> CustomerProfile:
        """
        生成单个客户的特征向量
        
        Args:
            current_day: 当前日期
            
        Returns:
            客户特征配置
        """
        lead_time = self._sample_lead_time(current_day)
        
        # 2. 目标入住日期 T_stay = CurrentDate + L
        target_date = current_day + lead_time
        
        # 3. 最高支付意愿 WTP_i ~ Normal(μ_adr, σ_adr)
        wtp_params = self.params.wtp_params

        wtp_mean = float(wtp_params.get('mean', 100.0))
        wtp_std = float(wtp_params.get('std', 30.0))

        by_season_weekday = wtp_params.get('by_season_weekday')
        if isinstance(by_season_weekday, dict):
            # WTP should align with the stay date rather than the booking date.
            _, season, is_weekend = self._calendar_features(target_date)

            seg = by_season_weekday.get(season) if isinstance(by_season_weekday, dict) else None
            if isinstance(seg, dict):
                stats = seg.get(is_weekend)
                if isinstance(stats, dict):
                    wtp_mean = float(stats.get('mean', wtp_mean))
                    wtp_std = float(stats.get('std', wtp_std))

        if wtp_std <= 0:
            wtp_std = 30.0
        wtp_mean, wtp_std = self._apply_wtp_perturbation(wtp_mean, wtp_std)
        wtp = self.rng.normal(wtp_mean, wtp_std)
        wtp = max(10.0, wtp)  # 确保不低于最低价
        
        # 4. 价格敏感度 β_i
        # 方案1：基于提前期的关联性
        # β_i = β_base + α * log(1 + L_i) + ε
        # 方案2：简单均匀分布（当前使用）
        beta_min, beta_max = self.params.beta_range
        price_sensitivity = float(self.rng.uniform(beta_min, beta_max))
        
        # 5. 客户类型（线上/线下）
        # 根据历史数据比例随机分配
        base_online_only_prob = float(self.params.customer_type_ratio[0])
        online_only_prob = self._apply_channel_online_only_prob(base_online_only_prob)
        customer_type = self.rng.choice(
            ['online_only', 'omnichannel'],
            p=[online_only_prob, 1.0 - online_only_prob],
        )
        
        return CustomerProfile(
            lead_time=lead_time,
            target_date=target_date,
            wtp=wtp,
            price_sensitivity=price_sensitivity,
            customer_type=customer_type
        )
    
    def simulate_day(self, 
                     price_online: float, 
                     price_offline: float) -> Dict:
        """
        模拟一天的酒店运营
        
        Args:
            price_online: 线上渠道报价
            price_offline: 线下渠道报价
            
        Returns:
            当日统计数据
        """
        anchor_reference_prices = self.build_daily_anchor_reference_prices()

        # 生成当日客户
        daily_customers = self.generate_daily_customers(self.current_day)
        
        # 统计变量
        new_bookings_online = 0
        new_bookings_offline = 0
        cancellations = 0
        room_marginal_cost = float(max(0.0, getattr(self.params, "room_marginal_cost", 0.0)))
        commission_rate = float(np.clip(getattr(self, "commission_rate", 0.0), 0.0, 1.0))
        revenue_online = 0.0
        revenue_offline = 0.0
        gross_revenue_hotel = 0.0
        hotel_marginal_cost = 0.0
        
        # 按day_offset统计预订信息（用于强化学习更新）
        bookings_online_by_offset = [0] * self.booking_window_days
        bookings_offline_by_offset = [0] * self.booking_window_days
        revenue_online_by_offset = [0.0] * self.booking_window_days
        revenue_offline_by_offset = [0.0] * self.booking_window_days
        
        # 客户决策阶段
        for customer in daily_customers:
            target_date = customer.profile.target_date
            days_ahead = target_date - self.current_day

            if not (0 <= days_ahead < self.booking_window_days):
                continue

            online_price = self.price_window_online[days_ahead]
            online_price_base = self.price_window_online_base[days_ahead]
            offline_price = self.price_window_offline[days_ahead]

            # 做出预订决策
            inventory_before = int(self.daily_available_rooms[target_date])
            if customer.make_booking_decision(
                online_price,
                offline_price,
                self.current_day,
                anchor_reference_prices,
            ):
                target_date = customer.booking_record.target_date
                
                # ✅ 正确的库存检查：检查目标日期的库存
                if self.daily_available_rooms[target_date] > 0:
                    # ✅ 扣减该日期的库存
                    self.daily_available_rooms[target_date] -= 1
                    
                    self.active_bookings.append(customer.booking_record)
                    self.booking_history.append(customer.booking_record)
                    self.total_bookings_count += 1
                    
                    # 统计总预订量
                    if customer.booking_record.customer_type == 'online':
                        new_bookings_online += 1
                        hotel_gross_online = float(online_price_base) * (1.0 - commission_rate)
                        hotel_net_online = hotel_gross_online - room_marginal_cost
                        revenue_online += hotel_net_online
                        gross_revenue_hotel += hotel_gross_online
                        hotel_marginal_cost += room_marginal_cost
                    else:
                        new_bookings_offline += 1
                        hotel_gross_offline = float(offline_price)
                        hotel_net_offline = hotel_gross_offline - room_marginal_cost
                        revenue_offline += hotel_net_offline
                        gross_revenue_hotel += hotel_gross_offline
                        hotel_marginal_cost += room_marginal_cost
                    
                    # 统计按day_offset分组的预订信息
                    if 0 <= days_ahead < self.booking_window_days:
                        if customer.booking_record.customer_type == 'online':
                            bookings_online_by_offset[days_ahead] += 1
                            revenue_online_by_offset[days_ahead] += hotel_net_online
                        else:
                            bookings_offline_by_offset[days_ahead] += 1
                            revenue_offline_by_offset[days_ahead] += hotel_net_offline
                    self._record_customer_utility_trace(
                        customer=customer,
                        current_day=self.current_day,
                        day_offset=days_ahead,
                        online_price=online_price,
                        online_price_base=online_price_base,
                        offline_price=offline_price,
                        inventory_before=inventory_before,
                        inventory_after=int(self.daily_available_rooms[target_date]),
                        final_status=f"booked_{customer.booking_record.customer_type}",
                    )
                else:
                    self._record_customer_utility_trace(
                        customer=customer,
                        current_day=self.current_day,
                        day_offset=days_ahead,
                        online_price=online_price,
                        online_price_base=online_price_base,
                        offline_price=offline_price,
                        inventory_before=inventory_before,
                        inventory_after=inventory_before,
                        final_status='rejected_inventory',
                    )
                # else: 该日期已满房，拒绝预订
            else:
                self._record_customer_utility_trace(
                    customer=customer,
                    current_day=self.current_day,
                    day_offset=days_ahead,
                    online_price=online_price,
                    online_price_base=online_price_base,
                    offline_price=offline_price,
                    inventory_before=inventory_before,
                    inventory_after=inventory_before,
                    final_status='rejected_threshold',
                )
            # 直接在simulate_day中创建booking_record，跳过效用函数
            #if not customer.has_booked and self.daily_available_rooms[target_date] > 0:
            #    customer.has_booked = True
            #    booking_record = BookingRecord(
            #        customer_id=customer.unique_id,
            #        booking_date=self.current_day,
            #        target_date=target_date,
            #        paid_price=price,
            #        wtp=customer.profile.wtp,
            #        price_sensitivity=customer.profile.price_sensitivity,
            #        customer_type=customer.profile.customer_type)
            #    customer.booking_record = booking_record
            #    
            #    target_date = customer.booking_record.target_date
                
                # ✅ 正确的库存检查：检查目标日期的库存
            #    if self.daily_available_rooms[target_date] > 0:
                # ✅ 扣减该日期的库存
            #        self.daily_available_rooms[target_date] -= 1
            #        
            #        self.active_bookings.append(customer.booking_record)
            #        self.booking_history.append(customer.booking_record)
                    
            #    if customer.profile.customer_type == 'online':
            #        new_bookings_online += 1
            #    else:
            #        new_bookings_offline += 1
                # else: 该日期已满房，拒绝预订

        
        # 取消评估阶段：遍历所有活跃预订
        #cancellation_refund = 0.0  # 记录取消订单的退款总额
        #active_bookings_copy = self.active_bookings.copy()
        #for booking in active_bookings_copy:
            # 找到对应的客户（如果还在系统中）
            # 简化：直接基于预订记录评估
            #days_until_checkin = booking.target_date - self.current_day
            
            #if days_until_checkin <= 0:
                # 已入住，移除活跃预订
            #    self.active_bookings.remove(booking)
            #    continue
            
            # ✅ 评估取消：使用价格窗口中对应日期的价格
            #days_ahead = booking.target_date - self.current_day
            #if 0 <= days_ahead < len(self.price_window_online):
            #    if booking.customer_type == 'online':
            #        current_price = self.price_window_online[days_ahead]
            #    else:
            #        current_price = self.price_window_offline[days_ahead]
            #else:
            #    # 超出窗口，使用默认价格
            #    current_price = price_online if booking.customer_type == 'online' else price_offline
            
            # 计算持有效用
            #satisfaction = booking.wtp - booking.paid_price
            #regret_coef = self.params.regret_coefficient
            #price_regret = regret_coef * max(0, booking.paid_price - current_price)
            #commitment_weight = self.params.commitment_weight
            #commitment_utility = commitment_weight / (days_until_checkin + 1)
            #shock_std = self.params.shock_std
            #daily_shock = np.random.normal(0, shock_std)
            
            #holding_utility = satisfaction - price_regret + commitment_utility + daily_shock
            
            # 取消决策
            #if holding_utility < 0 and not booking.is_canceled:
            #    booking.is_canceled = True
                # ✅ 释放该日期的库存
            #    self.daily_available_rooms[booking.target_date] += 1
                # ✅ 可退款政策：记录退款金额
            #    cancellation_refund += booking.paid_price
            #    self.active_bookings.remove(booking)
            #    cancellations += 1
        
        # 记录当日统计
        gross_revenue = gross_revenue_hotel
        #net_revenue = gross_revenue - cancellation_refund  # ✅ 净收益 = 新预订收益 - 取消退款
        net_revenue = revenue_online + revenue_offline
        bookings_by_day_offset = [
            {
                'day_offset': i,
                'bookings_online': bookings_online_by_offset[i],
                'bookings_offline': bookings_offline_by_offset[i],
                'revenue_online': revenue_online_by_offset[i],
                'revenue_offline': revenue_offline_by_offset[i],
            }
            for i in range(self.booking_window_days)
        ]

        daily_stat = {
            'day': self.current_day,
            'price_online': price_online,
            'price_offline': price_offline,
            'new_customers': len(daily_customers),
            'new_bookings_online': new_bookings_online,
            'new_bookings_offline': new_bookings_offline,
            'total_new_bookings': new_bookings_online + new_bookings_offline,
            'cancellations': cancellations,
            #'cancellation_refund': cancellation_refund,  # ✅ 记录退款金额
            'active_bookings': len(self.active_bookings),
            'revenue_online': revenue_online,
            'revenue_offline': revenue_offline,
            'gross_revenue': gross_revenue,  # ✅ 毛收益（新预订）
            'hotel_marginal_cost': hotel_marginal_cost,
            'total_revenue': net_revenue,  # ✅ 净收益（扣除退款后）
            'bookings_by_day_offset': bookings_by_day_offset,  # ✅ 按day_offset分组的预订信息
        }
        
        self.daily_stats.append(daily_stat)
        
        # 更新仿真日期
        self.current_day += 1
        
        # 收集数据
        self.datacollector.collect(self)
        
        return daily_stat
    
    def get_demand_prediction(self, 
                             price_online: float, 
                             price_offline: float,
                             num_simulations: int = 10) -> Tuple[float, float]:
        """
        获取需求预测（替代NGBoost的接口）
        
        通过多次蒙特卡洛模拟估计需求的均值和方差
        
        Args:
            price_online: 线上价格
            price_offline: 线下价格
            num_simulations: 模拟次数
            
        Returns:
            (预测需求均值, 预测需求方差)
        """
        # 保存当前状态
        original_day = self.current_day
        original_bookings = self.active_bookings.copy()
        original_seed = np.random.get_state()
        
        # 多次模拟
        demand_samples = []
        
        for _ in range(num_simulations):
            # 重置到当前状态
            self.current_day = original_day
            self.active_bookings = original_bookings.copy()
            
            # 模拟一天
            stat = self.simulate_day(price_online, price_offline)
            demand_samples.append(stat['total_new_bookings'])
            
            # 恢复状态（避免影响主仿真）
            self.current_day = original_day
            self.active_bookings = original_bookings.copy()
        
        # 恢复随机状态
        np.random.set_state(original_seed)
        
        # 计算均值和方差
        mean_demand = np.mean(demand_samples)
        var_demand = np.var(demand_samples)
        
        return mean_demand, var_demand
    
    def reset(self):
        """重置模型到初始状态"""
        self.current_day = 0
        self.active_bookings = []
        self.booking_history = []
        self.daily_stats = []
        self.customer_utility_trace = []
        self.total_customers_generated = 0
        self.total_bookings_count = 0
        self.total_cancellations_count = 0
        self._calendar_feature_cache = {}
        self.datacollector = self._build_datacollector()
    
    def get_statistics(self) -> pd.DataFrame:
        """
        获取统计数据
        
        Returns:
            每日统计数据DataFrame
        """
        return pd.DataFrame(self.daily_stats)

    def get_customer_utility_trace(self) -> pd.DataFrame:
        """返回当前episode的消费者效用轨迹。"""
        return pd.DataFrame(self.customer_utility_trace)
    
    def step(self):
        """
        模型的一步（由Mesa框架调用）
        """
        self.datacollector.collect(self)
