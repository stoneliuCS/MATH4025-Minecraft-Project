"""N-step replay buffer for SAC.

Wraps SB3's ReplayBuffer to accumulate n-step discounted returns before
storing transitions. This helps propagate sparse rewards (e.g. +100 for
collecting a log) back to the actions that caused them.
"""

from collections import deque

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class NStepReplayBuffer(ReplayBuffer):
    """ReplayBuffer that stores n-step returns instead of single-step transitions."""

    def __init__(self, *args, n_steps: int = 25, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        self._pending = deque()  # buffer of (obs, action, reward, next_obs, done)

    def add(self, obs, next_obs, action, reward, done, infos):
        self._pending.append((obs, next_obs, action, float(reward), bool(done)))

        # Only commit once we have n steps, or the episode ends
        if len(self._pending) < self.n_steps and not done:
            return

        # Compute n-step discounted return
        obs_0, _, action_0, _, _ = self._pending[0]
        n_step_reward = 0.0
        actual_next_obs = next_obs
        actual_done = done

        for i, (_, n_obs, _, r, d) in enumerate(self._pending):
            n_step_reward += (self.gamma ** i) * r
            actual_next_obs = n_obs
            actual_done = d
            if d:
                break

        super().add(obs_0, actual_next_obs, action_0, np.array([n_step_reward]),
                    np.array([actual_done]), infos)

        # Pop the oldest transition
        self._pending.popleft()

        # On episode end, flush remaining pending transitions
        if done:
            while self._pending:
                obs_i, _, action_i, _, _ = self._pending[0]
                n_step_reward = 0.0
                for j, (_, n_obs, _, r, d) in enumerate(self._pending):
                    n_step_reward += (self.gamma ** j) * r
                    actual_next_obs = n_obs
                    actual_done = d
                    if d:
                        break
                super().add(obs_i, actual_next_obs, action_i, np.array([n_step_reward]),
                            np.array([actual_done]), infos)
                self._pending.popleft()
