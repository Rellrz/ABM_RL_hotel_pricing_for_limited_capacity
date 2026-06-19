"""
OTA启发式补贴模块

将OTA从学习型Agent降维为外生启发式环境：
- 不再进行状态建模与策略学习
- 基于价格关系与提前预订天数(lead_time)实时计算补贴比例
"""

from typing import Dict, Optional

import numpy as np


class OTASubsidyHeuristic:
    """时间敏感 + 随机扰动 的OTA补贴启发式策略。"""

    def __init__(
        self,
        commission_rate: float = 0.15,
        r_max: float = 0.8,
        delta_max: float = 15.0,
        decay_lambda: float = 0.05,
        noise_std: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.commission_rate = float(commission_rate)
        self.r_max = float(r_max)
        self.delta_max = float(delta_max)
        self.decay_lambda = float(decay_lambda)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

        self.total_profit = 0.0
        self.total_commission = 0.0
        self.total_subsidy_cost = 0.0
        self.call_count = 0

    def get_subsidy(self, p_online_base: float, p_offline: float, lead_time: int) -> float:
        """
        计算补贴比例:
        1) Δ(t) = Δ_max * exp(-lambda * t)
        2) r_target = (P_on_base - P_off + Δ(t)) / (c * P_on_base)
        3) r_actual = clip(r_target + eps, 0, r_max), eps~N(0, noise_std^2)
        """
        p_online_base = float(p_online_base)
        p_offline = float(p_offline)
        t = max(0, int(lead_time))

        # 防止除零与无效价格
        if p_online_base <= 0.0 or self.commission_rate <= 0.0:
            return 0.0

        delta_t = self.delta_max * np.exp(-self.decay_lambda * t)
        r_target = (p_online_base - p_offline + delta_t) / (self.commission_rate * p_online_base)
        eps = float(self.rng.normal(0.0, self.noise_std)) if self.noise_std > 0 else 0.0
        r_actual = float(np.clip(r_target + eps, 0.0, self.r_max))

        self.call_count += 1
        return r_actual

    def calculate_profit(self, bookings_online: int, price_online_base: float, subsidy_ratio: float) -> float:
        """利润 = 佣金收入 - 补贴支出。"""
        bookings_online = int(max(0, bookings_online))
        price_online_base = float(price_online_base)
        subsidy_ratio = float(np.clip(subsidy_ratio, 0.0, self.r_max))

        commission_revenue = bookings_online * price_online_base * self.commission_rate
        subsidy_cost = commission_revenue * subsidy_ratio
        profit = commission_revenue - subsidy_cost

        self.total_profit += profit
        self.total_commission += commission_revenue
        self.total_subsidy_cost += subsidy_cost
        return float(profit)

    def get_statistics(self) -> Dict[str, float]:
        ratio = self.total_subsidy_cost / max(1e-8, self.total_commission)
        return {
            "total_profit": float(self.total_profit),
            "total_commission": float(self.total_commission),
            "total_subsidy_cost": float(self.total_subsidy_cost),
            "subsidy_ratio": float(ratio),
            "call_count": float(self.call_count),
            "avg_profit_per_episode": float(self.total_profit),
        }

    def save_model(self, file_name: str) -> str:
        from configs.config import PATH_CONFIG
        from datetime import datetime
        import json
        import os

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(PATH_CONFIG.models_dir, f"{file_name}_agent_{timestamp}.json")
        payload = {
            "type": "ota_heuristic",
            "commission_rate": self.commission_rate,
            "r_max": self.r_max,
            "delta_max": self.delta_max,
            "decay_lambda": self.decay_lambda,
            "noise_std": self.noise_std,
            "statistics": self.get_statistics(),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return save_path


# 向后兼容：旧代码若仍引用 OTAAgent 不会中断
class OTAAgent(OTASubsidyHeuristic):
    pass
