"""Behavioural Cloning regularization callback for SAC.

After each SAC gradient update, adds a BC loss term that penalizes the actor
for deviating from the pretrained policy. This prevents catastrophic forgetting
while still allowing SAC to improve on top of the pretrained weights.

L_total = L_SAC + lambda_bc * L_BC

where L_BC = -log_prob(pretrained_actions | current_policy, obs)
"""

import logging

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class BCRegularizationCallback(BaseCallback):
    """Applies a BC regularization loss after each SAC update.

    Samples a batch from the replay buffer, computes the log probability of
    the pretrained actions under the current actor, and applies a gradient
    step to keep the actor close to the pretrained policy.

    Args:
        pretrained_path: Path to the pretrained SAC zip file.
        lambda_bc: Weight of the BC loss relative to SAC loss.
        bc_batch_size: Number of transitions to sample for BC loss.
    """

    def __init__(
        self,
        pretrained_path: str,
        lambda_bc: float = 0.5,
        bc_batch_size: int = 64,
    ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.lambda_bc = lambda_bc
        self.bc_batch_size = bc_batch_size
        self._pretrained_actor = None

    def _on_training_start(self) -> None:
        from stable_baselines3 import SAC
        pretrained = SAC.load(self.pretrained_path)
        self._pretrained_actor = pretrained.policy.actor
        self._pretrained_actor.set_training_mode(False)
        self._pretrained_actor.to(self.model.device)
        logger.info(f"BC regularization loaded pretrained actor from {self.pretrained_path}")

    def _on_step(self) -> bool:
        # Only apply after learning has started and buffer has enough samples
        if self.model.num_timesteps < self.model.learning_starts:
            return True
        if self.model.replay_buffer.size() < self.bc_batch_size:
            return True

        replay_data = self.model.replay_buffer.sample(
            self.bc_batch_size, env=self.model._vec_normalize_env
        )

        obs = replay_data.observations
        if isinstance(obs, dict):
            obs = obs["obs"]

        # Get actions from pretrained actor
        with torch.no_grad():
            pretrained_actions, _ = self._pretrained_actor.action_log_prob(obs)

        # Compute log prob of pretrained actions under current actor
        mean, log_std, _ = self.model.policy.actor.get_action_dist_params(obs)
        dist = self.model.policy.actor.action_dist.proba_distribution(mean, log_std)
        bc_loss = -dist.log_prob(pretrained_actions).mean()

        # Apply BC gradient step
        self.model.policy.actor.optimizer.zero_grad()
        (self.lambda_bc * bc_loss).backward()
        self.model.policy.actor.optimizer.step()

        return True
