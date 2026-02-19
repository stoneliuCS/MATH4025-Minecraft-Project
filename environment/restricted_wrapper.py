import gym
import numpy as np

class RestrictedActionWrapper(gym.ActionWrapper):
  """Only allow movement actions with a simple 4-action discrete space."""

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
  yellow_penalty = -50.0          # One-time penalty for entering yellow
  distance_reward_scale = 1.0      # Small reward for progress
  step_penalty = -0.1              # Small penalty to encourage efficiency


  target_x = -6.5
  target_z = 193.5

  # Action mapping: index -> direction name
  ACTION_NAMES = ["forward", "back", "left", "right"]

  def __init__(self, env):
        super().__init__(env)
        # Simple discrete action space: 0=forward, 1=back, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)
        self.prev_distance = None
        self.touched_yellow = False
        self.touched_red = False
        self.yellow_zones_visited = set()
        self.was_in_yellow = False  # Track if agent was in yellow last step
        self.last_filtered_action = None  # Store last filtered action for debugging

  def reset(self, **kwargs):
        self.prev_distance = None
        self.touched_yellow = False
        self.touched_red = False
        self.prev_wood = 0
        self.yellow_zones_visited = set()
        self.was_in_yellow = False
        return self.env.reset(**kwargs)

  def action(self, action):
    """Convert discrete action index to dict for underlying environment.

    Action mapping:
      0 = forward
      1 = back
      2 = left
      3 = right
    """
    # Build dict with all movement keys set to 0
    filtered = {"forward": 0, "back": 0, "left": 0, "right": 0}

    # Set the selected action to 1
    action_name = self.ACTION_NAMES[action]
    filtered[action_name] = 1

    # Store for debugging
    self.last_filtered_action = action_name

    return filtered

  def _in_zone(self, x, z, zones):
    """Check if (x, z) is in any zone. Returns zone index or -1."""
    for i, zone in enumerate(zones):
        x_min, x_max, z_min, z_max = zone
        if x_min <= x <= x_max and z_min <= z <= z_max:
            return i
    return -1

  def step(self, action):
    # Convert discrete action index to dict before passing to underlying env
    converted_action = self.action(action)
    obs, reward, done, info = self.env.step(converted_action)
    wood_now = obs.get("inventory", {}).get("log", 0)
    reward += (wood_now - self.prev_wood) * 100
    self.prev_wood = wood_now

    
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
        
        # 2. Check for yellow zone - penalize only on ENTRY (transition from outside to inside)
        yellow_idx = self._in_zone(x, z, self.yellow_zones)
        in_yellow_now = yellow_idx >= 0

        if in_yellow_now and not self.was_in_yellow:
            reward += self.yellow_penalty
            self.yellow_zones_visited.add(yellow_idx)
            self.touched_yellow = True
            info['touched_yellow_wool'] = True

        self.was_in_yellow = in_yellow_now
        
        # 3. Check for red zone (goal)
        if self._in_zone(x, z, self.red_zones) >= 0:
            reward += self.red_reward
            self.touched_red = True
            done = True
            info['touched_red_wool'] = True
    
    return obs, reward, done, info
