from minerl.herobraine.env_spec import TranslationHandler
from minerl.herobraine.env_specs.basalt_specs import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing_extensions import override
from environment.ray import ObservationFromRay

DOC = """
This environment creates a very controlled, simple boxed world where the agent must navigate to find a simple cube.
"""

MAX_EPISODE_STEPS = 8000
MAX_REWARD_THRESHOLD = 100


class WorldEnvironment(HumanControlEnvSpec):
    def __init__(self, *args, world_path=None, **kwargs):
        # Allow callers to pass world_path directly, or via kwargs for backward compatibility.
        if world_path is None:
            world_path = kwargs.pop('world_path', None)

        self.world_path = world_path or 'simple'

        if 'name' not in kwargs:
            kwargs['name'] = 'NewWorld'
        super().__init__(
            *args,
            name=kwargs['name'],
            max_episode_steps=MAX_EPISODE_STEPS,
            reward_threshold=MAX_REWARD_THRESHOLD,
        )

    '''def create_mission_handlers(self):
        # Get default handlers
        mission_handlers = super().create_mission_handlers()

        # Add compass to slot 0
        mission_handlers.append(
            handlers.InventoryItem(
                slot=0,
                name="compass",
                quantity=1
            )
        )

        return mission_handlers'''
    

    @override
    def create_server_world_generators(self) -> list[Handler]:
        return []

    @override
    def create_agent_start(self) -> list[Handler]:

      import os
      return [
          handlers.LoadWorldAgentStart(self.world_path),
          handlers.GammaSetting(2.0),
          handlers.FOVSetting(70.0),
          handlers.FakeCursorSize(16),
          handlers.GuiScale(1),
      ]
    
    @override
    def create_rewardables(self) -> list[TranslationHandler]:
        return []

    @override
    def create_agent_handlers(self) -> list[Handler]:
      return []


    @override
    def create_server_quit_producers(self) -> list[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(MAX_EPISODE_STEPS * 50),
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
            handlers.ObservationFromLifeStats(),
            ObservationFromRay()
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
                                         
