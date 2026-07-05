from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy


class BetaDistribution(DiagGaussianDistribution):
    """Independent Beta distributions mapped from [0, 1] to bounded actions in [-1, 1]."""

    def __init__(
        self,
        action_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        min_concentration: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__(action_dim)
        self.low = float(low)
        self.high = float(high)
        self.min_concentration = float(min_concentration)
        self.epsilon = float(epsilon)
        self.alpha: Optional[th.Tensor] = None
        self.beta: Optional[th.Tensor] = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple[nn.Module, nn.Parameter]:
        del log_std_init
        concentration_net = nn.Linear(latent_dim, 2 * self.action_dim)
        dummy_log_std = nn.Parameter(th.zeros(self.action_dim), requires_grad=False)
        return concentration_net, dummy_log_std

    def proba_distribution(self, action_params: th.Tensor, log_std: th.Tensor) -> "BetaDistribution":
        del log_std
        raw_alpha, raw_beta = th.chunk(action_params, chunks=2, dim=1)
        min_concentration = max(self.min_concentration, self.epsilon)
        self.alpha = F.softplus(raw_alpha) + min_concentration + self.epsilon
        self.beta = F.softplus(raw_beta) + min_concentration + self.epsilon
        self.distribution = th.distributions.Beta(self.alpha, self.beta)
        return self

    def _action_to_unit_interval(self, actions: th.Tensor) -> th.Tensor:
        unit_actions = (actions - self.low) / max(self.high - self.low, self.epsilon)
        return unit_actions.clamp(self.epsilon, 1.0 - self.epsilon)

    def _unit_interval_to_action(self, unit_actions: th.Tensor) -> th.Tensor:
        return self.low + (self.high - self.low) * unit_actions

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        unit_actions = self._action_to_unit_interval(actions)
        base_log_prob = self.distribution.log_prob(unit_actions)
        transform_log_abs_det = np.log(max(self.high - self.low, self.epsilon))
        return sum_independent_dims(base_log_prob - transform_log_abs_det)

    def entropy(self) -> Optional[th.Tensor]:
        transform_log_abs_det = np.log(max(self.high - self.low, self.epsilon))
        return sum_independent_dims(self.distribution.entropy() + transform_log_abs_det)

    def sample(self) -> th.Tensor:
        unit_actions = self.distribution.rsample()
        return self._unit_interval_to_action(unit_actions).clamp(self.low, self.high)

    def mode(self) -> th.Tensor:
        if self.alpha is None or self.beta is None:
            raise RuntimeError("proba_distribution() must be called before using the distribution.")
        unit_mean = self.alpha / (self.alpha + self.beta).clamp_min(self.epsilon)
        unit_mode = (self.alpha - 1.0) / (self.alpha + self.beta - 2.0).clamp_min(self.epsilon)
        has_internal_mode = (self.alpha > 1.0) & (self.beta > 1.0)
        unit_action = th.where(has_internal_mode, unit_mode, unit_mean)
        return self._unit_interval_to_action(unit_action.clamp(0.0, 1.0))

    def log_prob_from_params(
        self,
        action_params: th.Tensor,
        log_std: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_params, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class BetaActorCriticPolicy(ActorCriticPolicy):
    """PPO actor-critic policy using a Beta action distribution on [-1, 1]."""

    def __init__(self, *args, **kwargs):
        self.beta_distribution_kwargs = {
            "low": float(kwargs.pop("action_low", -1.0)),
            "high": float(kwargs.pop("action_high", 1.0)),
            "min_concentration": float(kwargs.pop("beta_min_concentration", 1.0)),
            "epsilon": float(kwargs.pop("beta_epsilon", 1e-6)),
        }
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        self.action_dist = BetaDistribution(
            int(np.prod(self.action_space.shape)),
            **self.beta_distribution_kwargs,
        )
        super()._build(lr_schedule)

    def _get_action_dist_from_latent(self, latent_pi):
        action_params = self.action_net(latent_pi)
        if isinstance(self.action_dist, BetaDistribution):
            return self.action_dist.proba_distribution(action_params, self.log_std)
        return super()._get_action_dist_from_latent(latent_pi)
