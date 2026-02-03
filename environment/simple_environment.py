import os
from minerl.herobraine.env_spec import TranslationHandler
from minerl.herobraine.env_specs.basalt_specs import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing_extensions import override
from minerl.herobraine.hero.mc import INVERSE_KEYMAP

DOC = """
This environment creates a very controlled, simple boxed world where the agent must navigate to find a simple cube.
"""

MAX_EPISODE_STEPS = 8000
MAX_REWARD_THRESHOLD = 100


# Movement Reward Handler
class MovementReward(handlers.RewardHandler):
    """Reward for moving (based on distance traveled)."""
    
    def __init__(self, reward_per_block=0.1):
        self.last_pos = None
        self.reward_per_block = reward_per_block
    
    def to_string(self):
        return "movement_reward"
    
    def xml_template(self):
        return ""
    
    def from_universal(self, obs):
        if 'location' not in obs:
            return 0
        
        pos = obs['location']
        if self.last_pos is None:
            self.last_pos = pos
            return 0
        
        # Calculate distance moved (ignore y-axis for horizontal movement)
        distance = ((pos['x'] - self.last_pos['x']) ** 2 + 
                   (pos['z'] - self.last_pos['z']) ** 2) ** 0.5
        
        self.last_pos = pos
        return distance * self.reward_per_block  # Small reward for moving


class BoxedNavigationSimpleEnvironment(HumanControlEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'BoxedNavigation-v0'
        super().__init__(
            *args,
            name=kwargs['name'],
            max_episode_steps=MAX_EPISODE_STEPS,
            reward_threshold=MAX_REWARD_THRESHOLD,
        )


    @override
    def create_server_world_generators(self) -> list[Handler]:
        return []

    @override
    def create_agent_start(self) -> list[Handler]:
      world_path = os.path.join(os.path.dirname(__file__), "worlds", "simple.zip")
      return [
          handlers.LoadWorldAgentStart(world_path),
          handlers.AgentStartPlacement(0, 5, 0, yaw=45.0),
          handlers.GammaSetting(2.0),
          handlers.FOVSetting(70.0),
          handlers.FakeCursorSize(16),
          handlers.GuiScale(1),
      ]
    
    @override
    def create_rewardables(self) -> list[TranslationHandler]:
        """
        Reward structure:
        - Movement reward (encourages exploration)
        - Big reward for touching red_wool (target)
        - Penalty for touching yellow_wool (obstacle)
        """
        return [
            # Movement reward - encourages agent to move around
            MovementReward(reward_per_block=0.1),
            
            # Target: Big reward for touching red_wool block
            handlers.RewardForTouchingBlockType([
                {'type': 'red_wool', 'behaviour': 'onceOnly', 'reward': '100'},
            ]),
            
            # Penalty: Negative reward for touching yellow_wool
            handlers.RewardForTouchingBlockType([
                {'type': 'yellow_wool', 'behaviour': 'onceOnly', 'reward': '-25'},
            ]),
            
            # Reward when mission ends
            handlers.RewardForMissionEnd(0),
        ]

    @override
    def create_agent_handlers(self) -> list[Handler]:
      return [
          # End episode when agent touches red_wool (target)
          handlers.AgentQuitFromTouchingBlockType([
              'red_wool'
          ])
      ]


    @override
    def create_server_quit_producers(self) -> list[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(MAX_EPISODE_STEPS * 50),  # 50ms per tick
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    @override
    def create_server_decorators(self) -> list[Handler]:
        return []

    @override
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    @override
    def is_from_folder(self, folder: str) -> bool:
        return folder == 'simple'

    @override
    def get_docstring(self):
        return super().get_docstring()
    
    @override
    def create_actionables(self) -> list[TranslationHandler]:
        return super().create_actionables() 

    @override
    def create_observables(self) -> list[TranslationHandler]:
        return super().create_observables() + [
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats()
        ]

    @override
    def create_server_initial_conditions(self) -> list[Handler]:
        return [
                  handlers.SpawningInitialCondition(
                      allow_spawning=False
                  ),
                  handlers.TimeInitialCondition(
                      allow_passage_of_time=False,
                      start_time=6000
                  ),
        ]