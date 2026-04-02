import gym
import math
import numpy as np

class RLHFActionWrapper(gym.ActionWrapper):
 
  
    def __init__(self, env):
        super().__init__(env)
        
        self.action_space = gym.spaces.Discrete(11)
        self.prev_distance = None
        self.last_filtered_action = None  # Store last filtered action for debugging

        # keep track of some location stats 
        self.location_stat_history = {
            "pitch" : [],
            "yaw"  : [],
            "xpos" : [],
            "zpos" : [],
            "ypos" : [],
        }

    def reset(self, **kwargs):
        self.prev_distance = None
        return self.env.reset(**kwargs)
    
    def create_agent_start(self):
        return []


    def format_action(self, action_idx):
        """Convert discrete action index to dict for underlying environment.
        """
        # Build dict with all movement keys set to 0
        formatted = {
            "ESC" : 0
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
        if action_idx==10:
            formatted["attack"] = 1
        
        print(formatted)

        return formatted

    def step(self, action):
        # Convert discrete action index to dict before passing to underlying env
        converted_action = self.format_action(action)
        obs, reward, done, info = self.env.step(converted_action)

        if 'ray' in obs: 
            print(obs['ray'])
        
        if 'location_stats' in obs:
            x = obs['location_stats']['xpos']
            z = obs['location_stats']['zpos']

            self.location_stat_history['xpos'].append(obs['location_stats']['xpos'])
            self.location_stat_history['zpos'].append(obs['location_stats']['zpos'])
            self.location_stat_history['ypos'].append(obs['location_stats']['ypos'])
            self.location_stat_history['pitch'].append(obs['location_stats']['pitch'])
            self.location_stat_history['yaw'].append(obs['location_stats']['yaw'])

            reward = 0.0
            #info['xpos'] = x
            #info['zpos'] = z

            info['location_stat_history'] = self.location_stat_history

        return obs, reward, done, info



def average_pitch(info):
    return sum(info['location_stat_history']['pitch']) / len(info['location_stat_history']['pitch'])
def horizontal_distance_traveled(info):
    return math.sqrt(
        (info['location_stat_history']['xpos'][0] - info['location_stat_history']['xpos'][-1]) ** 2
        +(info['location_stat_history']['zpos'][0] - info['location_stat_history']['zpos'][-1]) ** 2
    )

def print_info(info):
    average_pitch = sum(info['location_stat_history']['pitch']) / len(info['location_stat_history']['pitch'])
    horizontal_distance_traveled = math.sqrt(
        (info['location_stat_history']['xpos'][0] - info['location_stat_history']['xpos'][-1]) ** 2
        +(info['location_stat_history']['zpos'][0] - info['location_stat_history']['zpos'][-1]) ** 2
    )
    print(f"start position (x,z): ({info['location_stat_history']['xpos'][0]},{info['location_stat_history']['zpos'][0]})")
    print(f"end position (x,z): ({info['location_stat_history']['xpos'][-1]},{info['location_stat_history']['zpos'][-1]})")
    print(f"average pitch: {average_pitch}")
    print(f"horizontal distance traveled: {horizontal_distance_traveled}")
 
