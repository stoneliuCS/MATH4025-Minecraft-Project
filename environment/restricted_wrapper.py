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
  red_reward = 1000.0              # Goal reward (large but not overwhelming)
  yellow_penalty = -100.0          # One-time penalty for entering yellow
  distance_reward_scale = 1.0      # Small reward for progress
  step_penalty = -0.1              # Small penalty to encourage efficiency


  target_x = -6.5
  target_z = 193.5
#   distance_reward_scale = 100.0

  def __init__(self, env):
        super().__init__(env)
        self.allowed_keys = {"forward", "back", "left", "right"}
        self.action_space = gym.spaces.Dict({
            key: space for key, space in env.action_space.spaces.items()
            if key in self.allowed_keys
        })
        self.prev_distance = None
        self.touched_yellow = False
        self.touched_red = False
        self.yellow_zones_visited = set()

  def reset(self, **kwargs):
        self.prev_distance = None
        self.touched_yellow = False
        self.touched_red = False
        self.yellow_zones_visited = set()
        return self.env.reset(**kwargs)

  def action(self, action):
    filtered = {}
    # First, filter to only allowed keys and zero out others
    for key, value in action.items():
        if key in self.allowed_keys:
            filtered[key] = value
        else:
            filtered[key] = np.zeros_like(value) if hasattr(value, 'shape') else 0
    
    # Ensure exactly one movement action is active at a time
    # Priority order: forward > back > left > right
    # This prevents the agent from moving in opposite directions simultaneously
    active_actions = []
    for key in ["forward", "back", "left", "right"]:
        if key in filtered:
            # Check if action is active (value > 0)
            val = filtered[key]
            # Handle both scalar and array values
            if isinstance(val, np.ndarray):
                is_active = bool(np.any(val > 0))
            else:
                is_active = bool(val > 0)
            if is_active:
                active_actions.append(key)
    
    # If no actions are active, force one (default to forward)
    if len(active_actions) == 0:
        # Set forward to active, matching the type of existing action values
        if "forward" in filtered:
            # Match the type of the existing value
            if isinstance(filtered["forward"], np.ndarray):
                filtered["forward"] = np.ones_like(filtered["forward"], dtype=filtered["forward"].dtype)
            else:
                # For scalar values, check if we need int or float
                # Try to infer from other action values
                sample_val = next(iter(filtered.values())) if filtered else 1
                if isinstance(sample_val, (int, np.integer)):
                    filtered["forward"] = 1
                else:
                    filtered["forward"] = 1.0
        else:
            # If forward doesn't exist, create it (default to 1 for binary actions)
            filtered["forward"] = 1
        active_actions = ["forward"]
    
    # If multiple actions are active, keep only the first one (by priority)
    if len(active_actions) > 1:
        # Keep the first active action (highest priority: forward > back > left > right)
        keep_action = active_actions[0]
        for key in ["forward", "back", "left", "right"]:
            if key in filtered and key != keep_action:
                # Zero out all other movement actions
                if isinstance(filtered[key], np.ndarray):
                    filtered[key] = np.zeros_like(filtered[key], dtype=filtered[key].dtype)
                else:
                    filtered[key] = 0
    
    return filtered

  def _in_zone(self, x, z, zones):
    """Check if (x, z) is in any zone. Returns zone index or -1."""
    for i, zone in enumerate(zones):
        x_min, x_max, z_min, z_max = zone
        if x_min <= x <= x_max and z_min <= z <= z_max:
            return i
    return -1

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    
    # Start with small step penalty to encourage efficiency
    reward = self.step_penalty
    
    if 'location_stats' in obs:
        x = obs['location_stats']['xpos']
        z = obs['location_stats']['zpos']
        
        # 1. Distance-based reward shaping
        current_distance = np.sqrt((x - self.target_x)**2 + (z - self.target_z)**2)
        
        if self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            reward += distance_delta * self.distance_reward_scale
        
        self.prev_distance = current_distance
        
        # 2. Check for yellow zone (one-time penalty per zone)
        yellow_idx = self._in_zone(x, z, self.yellow_zones)
        if yellow_idx >= 0 and yellow_idx not in self.yellow_zones_visited:
            reward += self.yellow_penalty
            self.yellow_zones_visited.add(yellow_idx)
            self.touched_yellow = True
            info['touched_yellow_wool'] = True
        
        # 3. Check for red zone (goal)
        if self._in_zone(x, z, self.red_zones) >= 0:
            reward += self.red_reward
            self.touched_red = True
            done = True
            info['touched_red_wool'] = True
    
    return obs, reward, done, info