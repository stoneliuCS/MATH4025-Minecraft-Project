from minerl.herobraine.env_spec import TranslationHandler
from minerl.herobraine.env_specs.basalt_specs import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing_extensions import override

DOC = """
This environment creates a very controlled, simple boxed world where the agent must navigate to find a simple cube.
"""

MAX_EPISODE_STEPS = 8000
MAX_REWARD_THRESHOLD = 100


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
        """
        No world generator needed - we load a pre-built world via LoadWorldAgentStart.
        """
        return []

    @override
    def create_agent_start(self) -> list[Handler]:

      import os
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
        return [
            handlers.RewardForTouchingBlockType([
                {'type': 'blue_wool', 'behaviour': 'onceOnly', 'reward': '50'},
            ]),
            handlers.RewardForMissionEnd(50)
        ]

    @override
    def create_agent_handlers(self) -> list[Handler]:
      return [
          handlers.AgentQuitFromTouchingBlockType([
              'blue_wool'
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
        """
        DrawingDecorator is not implemented in MCP-Reborn.
        Blocks must be pre-placed in the world file (boxed_arena.zip).

        The pre-built world should contain:
        - Stone walls from (-1,4,-1) to (10,6,10) forming a box
        - Smooth stone floor at y=4 from (0,4,0) to (9,4,9)
        - Red wool at (0,4,0) - spawn marker
        - Blue wool at (9,4,9) - goal marker
        """
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
                                         
