from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th

from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy


class TruncatedDiagGaussianDistribution(DiagGaussianDistribution):
    """Diagonal Gaussian policy with support truncated to the action bounds."""

    def __init__(
        self,
        action_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__(action_dim)
        self.low = float(low)
        self.high = float(high)
        self.epsilon = float(epsilon)
        self.mean_actions: Optional[th.Tensor] = None
        self.action_std: Optional[th.Tensor] = None

    def _adjust_std(self, mean_actions: th.Tensor, action_std: th.Tensor) -> th.Tensor:
        del mean_actions
        return action_std

    def proba_distribution(
        self,
        mean_actions: th.Tensor,
        log_std: th.Tensor,
    ) -> "TruncatedDiagGaussianDistribution":
        self.mean_actions = mean_actions
        base_std = th.ones_like(mean_actions) * log_std.exp()
        self.action_std = self._adjust_std(mean_actions, base_std).clamp_min(self.epsilon)
        self.distribution = th.distributions.Normal(self.mean_actions, self.action_std)
        return self

    def _standardized_bounds(self) -> tuple[th.Tensor, th.Tensor]:
        if self.mean_actions is None or self.action_std is None:
            raise RuntimeError("proba_distribution() must be called before using the distribution.")
        low = th.as_tensor(self.low, dtype=self.mean_actions.dtype, device=self.mean_actions.device)
        high = th.as_tensor(self.high, dtype=self.mean_actions.dtype, device=self.mean_actions.device)
        alpha = (low - self.mean_actions) / self.action_std
        beta = (high - self.mean_actions) / self.action_std
        return alpha, beta

    def _normalization_mass(self) -> th.Tensor:
        alpha, beta = self._standardized_bounds()
        standard_normal = th.distributions.Normal(th.zeros_like(alpha), th.ones_like(alpha))
        mass = standard_normal.cdf(beta) - standard_normal.cdf(alpha)
        return mass.clamp_min(self.epsilon)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        clipped_actions = actions.clamp(self.low + self.epsilon, self.high - self.epsilon)
        base_log_prob = self.distribution.log_prob(clipped_actions)
        log_mass = th.log(self._normalization_mass())
        return sum_independent_dims(base_log_prob - log_mass)

    def entropy(self) -> Optional[th.Tensor]:
        return None

    def sample(self) -> th.Tensor:
        alpha, beta = self._standardized_bounds()
        standard_normal = th.distributions.Normal(th.zeros_like(alpha), th.ones_like(alpha))
        lower_cdf = standard_normal.cdf(alpha)
        upper_cdf = standard_normal.cdf(beta)
        uniform = th.rand_like(lower_cdf)
        target_cdf = lower_cdf + uniform * (upper_cdf - lower_cdf)
        target_cdf = target_cdf.clamp(self.epsilon, 1.0 - self.epsilon)
        standard_sample = standard_normal.icdf(target_cdf)
        action = self.mean_actions + self.action_std * standard_sample
        return action.clamp(self.low, self.high)

    def mode(self) -> th.Tensor:
        if self.mean_actions is None:
            raise RuntimeError("proba_distribution() must be called before using the distribution.")
        return self.mean_actions.clamp(self.low, self.high)

    def log_prob_from_params(
        self,
        mean_actions: th.Tensor,
        log_std: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class ScaleAdjustedTruncatedDiagGaussianDistribution(TruncatedDiagGaussianDistribution):
    """Truncated Gaussian with the paper's boundary-aware scale discount."""

    def __init__(
        self,
        action_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        epsilon: float = 1e-6,
        scale_adjustment_k: float = 2.0,
        scale_adjustment_d_min: float = 0.01,
    ):
        super().__init__(action_dim=action_dim, low=low, high=high, epsilon=epsilon)
        self.scale_adjustment_k = float(scale_adjustment_k)
        self.scale_adjustment_d_min = float(scale_adjustment_d_min)

    def _adjust_std(self, mean_actions: th.Tensor, action_std: th.Tensor) -> th.Tensor:
        center = 0.5 * (self.low + self.high)
        half_range = 0.5 * (self.high - self.low)
        relative_distance = ((mean_actions - center).abs() / max(half_range, self.epsilon)).clamp(0.0, 1.0)
        k = max(self.scale_adjustment_k, self.epsilon)
        d_min = min(max(self.scale_adjustment_d_min, self.epsilon), 1.0)
        discount_shape = (1.0 - relative_distance.pow(k)).clamp_min(0.0).pow(1.0 / k)
        discount = discount_shape * (1.0 - d_min) + d_min
        return action_std * discount


class TruncatedGaussianActorCriticPolicy(ActorCriticPolicy):
    """PPO actor-critic policy using a bounded truncated Gaussian action distribution."""

    distribution_cls = TruncatedDiagGaussianDistribution

    def __init__(self, *args, **kwargs):
        self.distribution_kwargs = {
            "low": float(kwargs.pop("action_low", -1.0)),
            "high": float(kwargs.pop("action_high", 1.0)),
            "epsilon": float(kwargs.pop("truncated_epsilon", 1e-6)),
        }
        super().__init__(*args, **kwargs)
        self.action_dist = self.distribution_cls(
            int(np.prod(self.action_space.shape)),
            **self.distribution_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = th.tanh(self.action_net(latent_pi))
        if isinstance(self.action_dist, TruncatedDiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        return super()._get_action_dist_from_latent(latent_pi)


class ScaleAdjustedTruncatedGaussianActorCriticPolicy(TruncatedGaussianActorCriticPolicy):
    """PPO actor-critic policy using scale-adjusted truncated Gaussian actions."""

    distribution_cls = ScaleAdjustedTruncatedDiagGaussianDistribution

    def __init__(self, *args, **kwargs):
        self.scale_adjustment_k = float(kwargs.pop("scale_adjustment_k", 2.0))
        self.scale_adjustment_d_min = float(kwargs.pop("scale_adjustment_d_min", 0.01))
        super().__init__(*args, **kwargs)
        self.action_dist = self.distribution_cls(
            int(np.prod(self.action_space.shape)),
            scale_adjustment_k=self.scale_adjustment_k,
            scale_adjustment_d_min=self.scale_adjustment_d_min,
            **self.distribution_kwargs,
        )
