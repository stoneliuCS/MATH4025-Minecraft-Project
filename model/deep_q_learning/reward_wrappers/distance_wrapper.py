import gym
import numpy as np

class DistanceActionWrapper(gym.ActionWrapper):
 
  
  def __init__(self, env):
        super().__init__(env)
        
        self.action_space = gym.spaces.Discrete(10)
        self.prev_distance = None
        self.last_filtered_action = None  # Store last filtered action for debugging

  def reset(self, **kwargs):
        self.prev_distance = None
        return self.env.reset(**kwargs)

  def format_action(self, action_idx):
    """Convert discrete action index to dict for underlying environment.

    Action mapping:
      0 = forward
      1 = back
      2 = left
      3 = right
    """
    # Build dict with all movement keys set to 0
    formatted = {
        "forward": 0,
        "back": 0,
        "left": 0,
        "right": 0,
        "camera": [0,0]
    }

    if action_idx==0:
        formatted["forward"] = 1
    if action_idx==1:
        formatted["back"] = 1
    if action_idx==2:
        formatted["left"] = 1
    if action_idx==3:
        formatted["right"] = 1
    if action_idx==4:
        formatted["camera"] = [4,0]
    if action_idx==5:
        formatted["camera"] = [-4,0]
    if action_idx==6:
        formatted["camera"] = [0,4]
    if action_idx==7:
        formatted["camera"] = [0,-4]
    if action_idx==8:
        formatted["jump"] = 1
    if action_idx==9:
        formatted["sprint"] = 1
    
    print(formatted)

    return formatted

  def step(self, action):
    # Convert discrete action index to dict before passing to underlying env
    converted_action = self.format_action(action)
    obs, reward, done, info = self.env.step(converted_action)
    
    if 'location_stats' in obs:
        x = obs['location_stats']['xpos']
        z = obs['location_stats']['zpos']

        # reward is just the distance from spawn
        reward = np.sqrt(x**2 + z**2)

        info['xpos'] = x
        info['zpos'] = z

    return obs, reward, done, info
