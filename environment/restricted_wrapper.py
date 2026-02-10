import gym
import numpy as np
from gym import spaces

class RestrictedActionWrapper(gym.ActionWrapper):
<<<<<<< Updated upstream
    """Only allow movement actions, zero out everything else including camera."""
    def __init__(self, env):
=======
  """Only allow movement actions with a simple 4-action discrete space."""

  touched_yellow = False
  touched_red = False

  # Name (key) in the observation dict that should contain the block
  # the agent is currently standing on, e.g. "minecraft:yellow_wool".
  # Make sure your environment exposes this (or change the key here).
  block_obs_key = "block_in_feet"
  yellow_wool_id = "yellow_wool"
  red_wool_id = "red_wool"
  blue_wool_id = "blue_wool"

  # Legacy coordinate-based zones kept as a fallback when block info
  # is not available in observations.
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
>>>>>>> Stashed changes
        super().__init__(env)
        # Keep only movement keys in action space (NO camera)
        self.allowed_keys = {"forward", "back", "left", "right"}
        
        # Create a new action space with only allowed actions
        self.action_space = spaces.Dict({
            "forward": env.action_space["forward"],
            "back": env.action_space["back"],
            "left": env.action_space["left"],
            "right": env.action_space["right"]
        })
    
    def action(self, action):
        # Start with the full action space from the original env
        filtered = {}
        
        # Get all keys from the original action space
        for key in self.env.action_space.spaces.keys():
            if key in self.allowed_keys and key in action:
                filtered[key] = action[key]
            else:
                # Set to no-op (0 for discrete, zeros for continuous)
                original_space = self.env.action_space[key]
                if isinstance(original_space, spaces.Box):
                    filtered[key] = np.zeros(original_space.shape, dtype=original_space.dtype)
                else:
                    filtered[key] = 0
        
<<<<<<< Updated upstream
        return filtered
=======
        if self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            reward += distance_delta * self.distance_reward_scale
        
        self.prev_distance = current_distance
        
        # 2. Prefer block-based detection if the environment exposes the block
        # under the agent (e.g. \"minecraft:yellow_wool\"). Fallback to the
        # old coordinate-based zones if that information is missing.
        block_id_raw = obs.get(self.block_obs_key, None)

        if block_id_raw is not None:
            block_id = str(block_id_raw).lower()

            in_yellow_now = self.yellow_wool_id in block_id
            in_red_now = self.red_wool_id in block_id
            in_blue_now = self.blue_wool_id in block_id

            # Penalize yellow only on entry
            if in_yellow_now and not self.was_in_yellow:
                reward += self.yellow_penalty
                self.touched_yellow = True
                info['touched_yellow_wool'] = True

            self.was_in_yellow = in_yellow_now

            # Red wool is the goal
            if in_red_now:
                reward += self.red_reward
                self.touched_red = True
                done = True
                info['touched_red_wool'] = True

            # Blue wool is tracked but does not currently affect reward
            if in_blue_now:
                info['touched_blue_wool'] = True
        else:
            # Fallback: legacy hard-coded zones using (x, z) coordinates.
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
>>>>>>> Stashed changes
