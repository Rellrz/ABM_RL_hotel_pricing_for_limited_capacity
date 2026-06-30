from __future__ import annotations

import numpy as np

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class TanhActorCriticPolicy(ActorCriticPolicy):
    """PPO policy that keeps the standard SB3 architecture but uses tanh-squashed Gaussian actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dist = SquashedDiagGaussianDistribution(int(np.prod(self.action_space.shape)))

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, SquashedDiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        return super()._get_action_dist_from_latent(latent_pi)
