import gym
import numpy as np

class RestrictedActionWrapper(gym.ActionWrapper):
  """Only allow movement actions, zero out everything else."""

  touched_yellow = False
  touched_red = False
  # Zones expanded to cover full block area: block at (bx, bz) -> (bx, bx+1, bz, bz+1)
  yellow_zones = [
      (-6, -5, 189, 190), (-6, -5, 191, 192),
      (-5, -4, 189, 190), (-5, -4, 193, 194),
      (-4, -3, 189, 193),  # covers 189-192 range
      (-3, -2, 189, 190), (-3, -2, 192, 193),
      (-2, -1, 189, 190), (-2, -1, 192, 193),
      (-1, 0, 193, 194),
      (0, 1, 189, 190),
  ]
  red_zones = [(-7, -6, 193, 194)]
  blue_zones = [(0, 1, 187, 188)]
  yellow_reward = -1.0
  red_reward = 50000

  target_x = -6.5
  target_z = 193.5
  distance_reward_scale = 100.0

  def __init__(self, env):
      super().__init__(env)
      # Keep only movement keys in action space
      self.allowed_keys = {"forward", "back", "left", "right"}
      self.action_space = gym.spaces.Dict({
          key: space for key, space in env.action_space.spaces.items()
          if key in self.allowed_keys
      })
      self.prev_distance = None  # Track previous distance for reward shaping

  def reset(self, **kwargs):
      self.prev_distance = None
      self.touched_yellow = False
      self.touched_red = False
      return self.env.reset(**kwargs)

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

  def _in_zone(self, x, z, zones):
      """Check if (x, z) coordinates are within any of the specified zones.

      Args:
          x: X coordinate of the agent
          z: Z coordinate of the agent
          zones: List of zones, each defined as (x_min, x_max, z_min, z_max)

      Returns:
          True if the position is within any zone, False otherwise
      """
      for zone in zones:
          x_min, x_max, z_min, z_max = zone
          if x_min <= x <= x_max and z_min <= z <= z_max:
              return True
      return False

  def step(self, action):
    obs, reward, done, info = self.env.step(action)

    # Get agent position from observation
    if 'location_stats' in obs:
        x = obs['location_stats']['xpos']
        z = obs['location_stats']['zpos']

        current_distance = np.sqrt((x - self.target_x)**2 + (z - self.target_z)**2)
        if self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            reward += distance_delta * self.distance_reward_scale
        self.prev_distance = current_distance

        if self._in_zone(x, z, self.yellow_zones):
            reward += self.yellow_reward
            self.touched_yellow = True
            info['touched_yellow_wool'] = True

        if self._in_zone(x, z, self.red_zones):
            reward += self.red_reward
            self.touched_red = True
            done = True
            info['touched_red_wool'] = True

    return obs, reward, done, info
