import gym
import numpy as np
from gym import spaces

class RestrictedActionWrapper(gym.ActionWrapper):
    """Only allow movement actions, zero out everything else including camera."""
    def __init__(self, env):
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
        
        return filtered