import gym
import numpy as np

class RestrictedActionWrapper(gym.ActionWrapper):
  """Only allow movement actions, zero out everything else."""

  def __init__(self, env):
      super().__init__(env)
      # Keep only movement keys in action space
      self.allowed_keys = {"forward", "back", "left", "right", "camera"}

  def action(self, action):
      # Zero out any non-movement actions
      filtered = {}
      for key, value in action.items():
          if key in self.allowed_keys:
              filtered[key] = value
          else:
              # Set to no-op (0)
              filtered[key] = np.zeros_like(value) if hasattr(value, 'shape') else 0
      return filtered
