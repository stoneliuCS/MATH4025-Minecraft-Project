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
            "ray_distance" : [],
            "ray_in_range" : [],
            
            # times when the agent is mining wood
            'mining_wood' : [],

            # times when the agent is looking at wood
            'acacia_leaves' : [],
            'acacia_log' : [],
            'birch_leaves' : [],
            'birch_log' : [],
            'dark_oak_leaves' : [],
            'dark_oak_log' : [],
            'jungle_leaves' : [],
            'jungle_log' : [],
            'oak_leaves' : [],
            'oak_log' : [],
            'spruce_leaves' : [],
            'spruce_log' : [],

            # record the index of the action which the agent chose in this time step
            'action_idx' : []
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

        # save the action
        self.location_stat_history['action_idx'].append(action)

        if 'ray' in obs: 
            # save the ray data in a way that is easy to use
            self.location_stat_history['ray_distance'].append(obs['ray']['ray_data']['distance'])
            self.location_stat_history['ray_in_range'].append(obs['ray']['ray_data']['in_range'])

            print(f"facing oak wood: {obs['ray']['ray_data']['type']['oak_log']}")
            
            self.location_stat_history['acacia_log'].append(obs['ray']['ray_data']['type']['acacia_log'])
            self.location_stat_history['birch_log'].append(obs['ray']['ray_data']['type']['birch_log'])
            self.location_stat_history['dark_oak_log'].append(obs['ray']['ray_data']['type']['dark_oak_log'])
            self.location_stat_history['jungle_log'].append(obs['ray']['ray_data']['type']['jungle_log'])
            self.location_stat_history['oak_log'].append(obs['ray']['ray_data']['type']['oak_log'])
            self.location_stat_history['spruce_log'].append(obs['ray']['ray_data']['type']['spruce_log'])

            # see when the agent is mining wood: 
            self.location_stat_history['mining_wood'].append(
                obs['ray']['ray_data']['mine_block']['acacia_log']
                +obs['ray']['ray_data']['mine_block']['birch_log']
                +obs['ray']['ray_data']['mine_block']['dark_oak_log']
                +obs['ray']['ray_data']['mine_block']['jungle_log']
                +obs['ray']['ray_data']['mine_block']['oak_log']
                +obs['ray']['ray_data']['mine_block']['spruce_log']
            )

        
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
    return sum(info['pitch']) / len(info['pitch'])
def horizontal_distance_traveled(info):
    return math.sqrt(
        (info['xpos'][0] - info['xpos'][-1]) ** 2
        +(info['zpos'][0] - info['zpos'][-1]) ** 2
    )
def get_looking_at_wood_history(info):
    # return a list that is a vector sum of each history vector for looking at wood
    array = np.array(info['acacia_log']) + np.array(info['birch_log']) + np.array(info['dark_oak_log']) + np.array(info['jungle_log']) + np.array(info['oak_log']) + np.array(info['spruce_log'])
    #print(f"sum of looking at wood history: {sum(array)}")
    return array
def fraction_of_time_looking_at_wood(info):
    # calculate the fraction of the time steps that the agent spent looking at wood
    _sum = sum(info['acacia_log']) + sum(info['birch_log']) + sum(info['dark_oak_log']) + sum(info['jungle_log']) + sum(info['oak_log']) + sum(info['spruce_log'])
    return _sum / len(info['oak_log'])
def fraction_of_time_attacking_wood(info):
    # figure out how much time the agent spent attacking while looking at wood
    # all of the time that the agent spent looking at wood
    looking_at_wood = get_looking_at_wood_history(info)
    # all of the times that the agent was attacking
    attacking = np.array([1 if x == 10 else 0 for x in info['action_idx']])
    #return the fraction
    return (looking_at_wood @ attacking) / len(looking_at_wood)
def fraction_of_time_moving_towards_wood(info):
    # how much time the agent was moving forward and looking at wood
    looking_at_wood = get_looking_at_wood_history(info)
    forward = np.array([1 if x == 0 else 0 for x in info['action_idx']])
    return (looking_at_wood @ forward) / len(looking_at_wood)

def get_preference_metrics(info):
    return {
        "average pitch" : average_pitch(info),
        "horizontal distance traveled" : horizontal_distance_traveled(info),
        "fraction of time looking at wood" : fraction_of_time_looking_at_wood(info),
        "fraction of time attacking wood" : fraction_of_time_attacking_wood(info),
        "fraction of time moving towards wood" : fraction_of_time_moving_towards_wood(info)
    }

def print_info(info_a, info_b):
    print("="*30)
    # calculate all of the preference metrics
    metrics_a = get_preference_metrics(info_a)
    metrics_b = get_preference_metrics(info_b)

    # compare each metric
    for metric in metrics_a.keys():
        print(f"{metric}:")
        print(f"segment 1: {metrics_a[metric]}")
        print(f"segment 2: {metrics_b[metric]}")
    print("="*30)
 
