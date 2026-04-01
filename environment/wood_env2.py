import logging
import random
import gym
import numpy as np
import cv2
import gym.spaces
logger = logging.getLogger(__name__)
from minerl.herobraine.env_spec import TranslationHandler
from minerl.herobraine.env_specs.basalt_specs import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing_extensions import override
from .ray import ObservationFromRay
from collections import deque


import minerl.herobraine.hero.handlers as handlers
MAX_EPISODE_STEPS = 2000
MAX_REWARD_THRESHOLD = 100
FRAME_SIZE = 64
CAMERA_MAX_ANGLE = 5.0
ACTION_DIM = 8  # 2 camera + 5 discrete (forward, back, left, right, attack)
LOG_ITEMS = ["oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log"]
MINECRAFT_LOG_ITEMS = ["minecraft:oak_log", "minecraft:spruce_log", "minecraft:birch_log", "minecraft:jungle_log", "minecraft:acacia_log", "minecraft:dark_oak_log"]
TREECHOP_WORLD_GENERATOR_OPTIONS = """{"coordinateScale":684.412,"heightScale":684.412,"lowerLimitScale":512.0,"upperLimitScale":512.0,"depthNoiseScaleX":200.0,"depthNoiseScaleZ":200.0,"depthNoiseScaleExponent":0.5,"mainNoiseScaleX":80.0,"mainNoiseScaleY":160.0,"mainNoiseScaleZ":80.0,"baseSize":8.5,"stretchY":12.0,"biomeDepthWeight":1.0,"biomeDepthOffset":0.0,"biomeScaleWeight":1.0,"biomeScaleOffset":0.0,"seaLevel":1,"useCaves":false,"useDungeons":false,"dungeonChance":8,"useStrongholds":false,"useVillages":false,"useMineShafts":false,"useTemples":false,"useMonuments":false,"useMansions":false,"useRavines":false,"useWaterLakes":false,"waterLakeChance":4,"useLavaLakes":false,"lavaLakeChance":80,"useLavaOceans":false,"fixedBiome":2,"biomeSize":4,"riverSize":1,"dirtSize":33,"dirtCount":10,"dirtMinHeight":0,"dirtMaxHeight":256,"gravelSize":33,"gravelCount":8,"gravelMinHeight":0,"gravelMaxHeight":256,"graniteSize":33,"graniteCount":10,"graniteMinHeight":0,"graniteMaxHeight":80,"dioriteSize":33,"dioriteCount":10,"dioriteMinHeight":0,"dioriteMaxHeight":80,"andesiteSize":33,"andesiteCount":10,"andesiteMinHeight":0,"andesiteMaxHeight":80,"coalSize":17,"coalCount":20,"coalMinHeight":0,"coalMaxHeight":128,"ironSize":9,"ironCount":20,"ironMinHeight":0,"ironMaxHeight":64,"goldSize":9,"goldCount":2,"goldMinHeight":0,"goldMaxHeight":32,"redstoneSize":8,"redstoneCount":8,"redstoneMinHeight":0,"redstoneMaxHeight":16,"diamondSize":8,"diamondCount":1,"diamondMinHeight":0,"diamondMaxHeight":16,"lapisSize":7,"lapisCount":1,"lapisCenterHeight":16,"lapisSpread":16}"""



"""
Randomized Environment
"""

class SafeMineRLWrapper(gym.Wrapper):
    def step(self, action):
        try:
            obs, reward, done, info = self.env.step(action)

            # MineRL sometimes signals error via info dict
            if info is not None and "error" in info:
                print("MineRL error detected in info. Forcing episode termination.")
                return self.env.reset(), 0.0, True, info

            return obs, reward, done, info

        except Exception as e:
            print("MineRL step crashed:", e)
            obs = self.env.reset()
            return obs, 0.0, True, {"error": str(e)}

    def reset(self, **kwargs):
        try:
            return self.env.reset(**kwargs)
        except Exception as e:
            print("MineRL reset crashed:", e)

            # HARD RESET strategy
            import time
            time.sleep(2)

            try:
                self.env.close()
            except:
                pass

            raise e  # let outer system recreate env

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


class RenderWrapper(gym.Wrapper):  # pyright: ignore[reportPrivateImportUsage]
    """Calls env.render() every step so the Minecraft GUI stays updated."""

    def step(self, action):
        self.env.render()
        return self.env.step(action)
    
class LookAtWoodRewardWrapper(gym.Wrapper):

    LOOK_REWARD = 0.005  # small shaping reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs_ray = obs.get("ray", {}).get("ray_data", {})
        # print(f"Ray observation: {obs_ray}")

        # depending on your schema, this might be:
        # obs_ray["type"] OR obs_ray["hit_type"] + separate type
        # block = obs_ray.get("type")
        in_range = obs_ray.get("in_range", 0)

        if any(obs_ray["type"].values()) and in_range == 1:
            reward += self.LOOK_REWARD
            # optional debug
            # print(f"[LOOK] +{self.LOOK_REWARD} for looking at {obs_ray['type']}")
        # elif block is not None:
        #     # optional debug for looking at other blocks
        #     print(f"[LOOK] No reward for looking at {block} (in_range={in_range})")
        #     pass

        return obs, reward, done, info
    
class MineBlockRewardWrapper(gym.Wrapper):

    LOG_ITEMS = {
        "oak_log", "spruce_log", "birch_log",
        "jungle_log", "acacia_log", "dark_oak_log"
    }

    LEAF_ITEMS = {
        "oak_leaves", "spruce_leaves", "birch_leaves",
        "jungle_leaves", "acacia_leaves", "dark_oak_leaves"
    }

    PENALTY_ITEMS = {
        "dirt", "grass_block"
    }

    LOG_REWARD = 5.0
    LEAF_REWARD = 0.001
    DIRT_PENALTY = 0.01
    GRASS_PENALTY = 0.01

    def __init__(self, env):
        super().__init__(env)
        self.prev_mine_counts = {}

    def reset(self, **kwargs):
        self.prev_mine_counts = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward -= 0.001

        obs_ray = obs.get("ray", {}).get("ray_data", {})
        # print(f"Ray observation: {obs_ray.get('hit_type')}")
        mined = obs_ray.get("mine_block")
        # print(f"mine_block observation: {mined}")

        if isinstance(mined, dict):
            for block_name, count in mined.items():

                prev_count = self.prev_mine_counts.get(block_name, 0)
                delta = count - prev_count

                if delta > 0:

                    if block_name in self.LOG_ITEMS:
                        bonus = self.LOG_REWARD * delta
                        reward += bonus
                        print(f"+{bonus:.2f} for {count}x {block_name}")

                    elif block_name in self.LEAF_ITEMS:
                        bonus = self.LEAF_REWARD * delta
                        reward += bonus
                        print(f"+{bonus:.2f} for {count}x {block_name}")

                    elif block_name in self.PENALTY_ITEMS:
                        penalty = self.DIRT_PENALTY * delta
                        if block_name == "grass_block":
                            penalty = self.GRASS_PENALTY * delta
                        reward -= penalty
                        print(f"-{penalty:.2f} for {count}x {block_name}")

            self.prev_mine_counts = mined.copy()

        return obs, reward, done, info
    
class StickyAttackWrapper(gym.Wrapper):
    def __init__(self, env, sticky_ticks=10):
        super().__init__(env)
        self.sticky_ticks = 5
        self.attack_counter = 0

    def step(self, action):
        if action["attack"] == 1:
            self.attack_counter = self.sticky_ticks

        if self.attack_counter > 0:
            action["attack"] = 1
            self.attack_counter -= 1

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
        noop["attack"] = int(action[6] > .25)
        noop["jump"] = int(action[7] > .95)
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
        seed = random.randint(0, 2**31 - 1)

        print(f"[GatherWood] Using world seed: {seed}")

        return [
            handlers.DefaultWorldGenerator(
                force_reset=True,
                # generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS
            )
        ]
        # return [handlers.DefaultWorldGenerator(force_reset=True)]

    @override
    def create_agent_start(self) -> list[Handler]:

    #   import os
    #   world_path = os.path.join(os.path.dirname(__file__), "worlds", "getwood.zip")
      return [
        #   handlers.LoadWorldAgentStart(world_path),
          handlers.SimpleInventoryAgentStart([
    {'type': 'diamond_axe', 'quantity': 64}
]),
        
          handlers.GammaSetting(2.0),
          handlers.FOVSetting(70.0),
          handlers.FakeCursorSize(16),
          handlers.GuiScale(1),
          handlers.PreferredSpawnBiome("taiga"),
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
            ObservationFromRay(),
        ]

    @override
    def create_server_initial_conditions(self) -> list[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]
                                         
