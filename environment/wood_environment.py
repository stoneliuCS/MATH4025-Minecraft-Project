import gym
import numpy as np
import cv2
import gym.spaces
from minerl.herobraine.env_spec import TranslationHandler
from minerl.herobraine.env_specs.basalt_specs import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing_extensions import override


MAX_EPISODE_STEPS = 8000
MAX_REWARD_THRESHOLD = 100
FRAME_SIZE = 64
CAMERA_MAX_ANGLE = 10.0
ACTION_DIM = 7  # 2 camera + 5 discrete (forward, back, left, right, attack)
LOG_ITEMS = ["oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log"]

"""
Randomized Environment
"""

class PovImageWrapper(gym.ObservationWrapper):  # pyright: ignore[reportPrivateImportUsage]
  """Extract 'pov' from MineRL Dict obs, resize, and return as (C, H, W) uint8 image.

  SB3's CnnPolicy expects channel-first image observations with pixel values in [0, 255].
  """

  def __init__(self, env):
      super().__init__(env)
      self.observation_space = gym.spaces.Box(
          low=0, high=255,
          shape=(3, FRAME_SIZE, FRAME_SIZE),
          dtype=np.uint8,
      )

  def observation(self, observation):
      pov = observation["pov"] if isinstance(observation, dict) else observation
      img = cv2.resize(pov, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
      # (H, W, C) -> (C, H, W)
      return np.transpose(img, (2, 0, 1)).astype(np.uint8)

  def reset(self, **kwargs):
      obs = self.env.reset(**kwargs)
      return self.observation(obs)

  def step(self, action):
      obs, reward, done, info = self.env.step(action)
      return self.observation(obs), reward, done, info

class LogRewardWrapper(gym.Wrapper):  # pyright: ignore[reportPrivateImportUsage]
    """Compute reward from inventory log count changes.

    The built-in RewardForCollectingItems silently returns 0 when the
    Malmo diff observation is missing.  This wrapper reads the inventory
    observation directly and rewards +1 per new log collected.
    """

    def __init__(self, env, reward_per_log: float = 1.0):
        super().__init__(env)
        self.reward_per_log = reward_per_log
        self._prev_logs = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_logs = self._get_log_count(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        cur_logs = self._get_log_count(obs)
        log_diff = cur_logs - self._prev_logs
        if log_diff > 0:
            reward += log_diff * self.reward_per_log
        self._prev_logs = cur_logs
        return obs, reward, done, info

    @staticmethod
    def _get_log_count(obs) -> int:
        if isinstance(obs, dict) and "inventory" in obs:
            inv = obs["inventory"]
            if isinstance(inv, dict):
                return sum(int(inv.get(item, 0)) for item in LOG_ITEMS)
        return 0


class StickyAttackWrapper(gym.Wrapper):  # pyright: ignore[reportPrivateImportUsage]
    """Keep attack held for a minimum number of ticks once activated.

    When the agent starts attacking, the attack key stays held for at least
    `sticky_ticks` steps (even if the agent tries to release it), and camera
    movement is dampened so the cursor stays on the block being mined.
    Other actions (movement) update normally every tick.
    """

    def __init__(self, env, sticky_ticks: int = 15):
        super().__init__(env)
        self.sticky_ticks = sticky_ticks
        self._attack_counter = 0

    def step(self, action):
        if isinstance(action, dict):
            if action.get("attack", 0) == 1:
                self._attack_counter = self.sticky_ticks
            if self._attack_counter > 0:
                action["attack"] = 1
                self._attack_counter -= 1
        return self.env.step(action)

    def reset(self, **kwargs):
        self._attack_counter = 0
        return self.env.reset(**kwargs)


class RenderWrapper(gym.Wrapper):  # pyright: ignore[reportPrivateImportUsage]
    """Calls env.render() every step so the Minecraft GUI stays updated."""

    def step(self, action):
        self.env.render()
        return self.env.step(action)


class ActionWrapper(gym.ActionWrapper):  # pyright: ignore[reportPrivateImportUsage]
    """Map a 7-dim continuous vector in [-1, 1] to a MineRL action dict.

    Layout:
        [0] camera pitch   — scaled to [-CAMERA_MAX_ANGLE, CAMERA_MAX_ANGLE]
        [1] camera yaw     — scaled to [-CAMERA_MAX_ANGLE, CAMERA_MAX_ANGLE]
        [2] forward        — > 0 => 1, else 0
        [3] back           — > 0 => 1, else 0
        [4] left           — > 0 => 1, else 0
        [5] right          — > 0 => 1, else 0
        [6] attack (punch) — > 0 => 1, else 0
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict:
        noop = self.env.action_space.noop()  # pyright: ignore[reportAttributeAccessIssue]
        noop["camera"] = np.array([
            action[0] * CAMERA_MAX_ANGLE,
            action[1] * CAMERA_MAX_ANGLE,
        ], dtype=np.float32)
        noop["forward"] = int(action[2] > 0)
        noop["back"] = int(action[3] > 0)
        noop["left"] = int(action[4] > 0)
        noop["right"] = int(action[5] > 0)
        noop["attack"] = int(action[6] > 0)
        return noop

    def reverse_action(self, action):
        raise NotImplementedError


class GatherWoodEnvironment(HumanControlEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'GatherWood-v0'
        super().__init__(
            *args,
            name=kwargs['name'],
            max_episode_steps=MAX_EPISODE_STEPS,
            reward_threshold=MAX_REWARD_THRESHOLD,
        )


    @override
    def create_server_world_generators(self) -> list[Handler]:
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    @override
    def create_agent_start(self) -> list[Handler]:
      return [
          handlers.GammaSetting(2.0),
          handlers.FOVSetting(70.0),
          handlers.FakeCursorSize(16),
          handlers.GuiScale(1),
          handlers.PreferredSpawnBiome("forest"),
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
        return sum(rewards) >= self.reward_threshold # pyright: ignore[reportOperatorIssue]

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
        ]

    @override
    def create_server_initial_conditions(self) -> list[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]
                                         
