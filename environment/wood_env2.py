import logging
import gym
import numpy as np
import cv2

logger = logging.getLogger(__name__)
from minerl.herobraine.env_spec import TranslationHandler
from minerl.herobraine.env_specs.basalt_specs import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing_extensions import override
from environment.ray import ObservationFromRay as RayObservation

MAX_EPISODE_STEPS = 1000
MAX_REWARD_THRESHOLD = 100
FRAME_SIZE = 64
CAMERA_MAX_ANGLE = 5.0
ACTION_DIM = 8  # 2 camera + 5 discrete (forward, back, left, right, attack)
LOG_ITEMS = [
    "oak_log",
    "spruce_log",
    "birch_log",
    "jungle_log",
    "acacia_log",
    "dark_oak_log",
]


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
            return self.env.reset(**kwargs)


class PovImageWrapper(gym.ObservationWrapper):  # pyright: ignore[reportPrivateImportUsage]
    """Extract 'pov' from MineRL Dict obs, resize, and return as (C, H, W) uint8 image.

    SB3's CnnPolicy expects channel-first image observations with pixel values in [0, 255].
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
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


class MineBlockRewardWrapper(gym.Wrapper):
    LOG_ITEMS = {
        "oak_log",
        "spruce_log",
        "birch_log",
        "jungle_log",
        "acacia_log",
        "dark_oak_log",
    }

    LEAF_ITEMS = {
        "oak_leaves",
        "spruce_leaves",
        "birch_leaves",
        "jungle_leaves",
        "acacia_leaves",
        "dark_oak_leaves",
    }

    PENALTY_ITEMS = {"dirt", "grass_block"}

    LOG_REWARD = 1.0
    LEAF_REWARD = 0.005
    DIRT_PENALTY = 0.002
    GRASS_PENALTY = 0.002
    ATTACK_LOG_REWARD = 0.05
    ATTACK_NONWOOD_PENALTY = 0.0001
    TIME_PENALTY = 0.0005

    def __init__(self, env):
        super().__init__(env)
        self.prev_mine_counts = {}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        ray_data = obs.get("ray", {}).get("ray_data", {}) if isinstance(obs, dict) else {}
        mine_block = ray_data.get("mine_block", {})
        self.prev_mine_counts = dict(mine_block) if isinstance(mine_block, dict) else {}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward -= self.TIME_PENALTY

        ray_data = obs.get("ray", {}).get("ray_data", {})
        mined = ray_data.get("mine_block")
        print(f"[MineBlock] mine_block={mined}")

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

        # Intermediate shaping: reward attacking while looking at a log in range
        ray_los = ray_data.get("type", {})
        looking_at_log = any(ray_los.get(k, 0) for k in ("oak_log", "birch_log", "spruce_log"))
        if isinstance(action, dict):
            attack_active = action.get("attack", 0) == 1
        else:
            attack_active = action[6] > 0.25  # numpy array from SAC policy
        if looking_at_log and ray_data.get("in_range") and attack_active:
            reward += self.ATTACK_LOG_REWARD
            print(f"[Reward] +{self.ATTACK_LOG_REWARD} attacking log")

        # Penalty for attacking anything that isn't a log or leaves
        if attack_active and ray_data.get("in_range") and not looking_at_log:
            looking_at_leaves = any(ray_data.get("type", {}).get(k, 0) for k in self.LEAF_ITEMS if k in ray_data.get("type", {}))
            if not looking_at_leaves:
                reward -= self.ATTACK_NONWOOD_PENALTY
                print(f"[Reward] -{self.ATTACK_NONWOOD_PENALTY} attacking non-wood")

        return obs, reward, done, info


class StickyAttackWrapper(gym.Wrapper):
    def __init__(self, env, sticky_ticks=10):
        super().__init__(env)
        self.sticky_ticks = sticky_ticks
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
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict:
        noop = self.env.action_space.noop()  # pyright: ignore[reportAttributeAccessIssue]
        noop["camera"] = np.array(
            [
                action[0] * CAMERA_MAX_ANGLE,
                action[1] * CAMERA_MAX_ANGLE,
            ],
            dtype=np.float32,
        )
        noop["forward"] = int(action[2] > 0)
        noop["back"] = int(action[3] > 0)
        noop["left"] = int(action[4] > 0)
        noop["right"] = int(action[5] > 0)
        noop["attack"] = int(action[6] > 0.25)
        noop["jump"] = int(action[7] > 0.95)
        return noop

    def reverse_action(self, action):
        raise NotImplementedError


class GatherWoodEnvironment(HumanControlEnvSpec):
    def __init__(self, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "GatherWood-v0"
        super().__init__(
            *args,
            name=kwargs["name"],
            max_episode_steps=MAX_EPISODE_STEPS,
            reward_threshold=MAX_REWARD_THRESHOLD,
        )

    @override
    def create_server_world_generators(self) -> list[Handler]:
        return [
            handlers.DefaultWorldGenerator(
                force_reset=True,
            )
        ]

    @override
    def create_agent_start(self) -> list[Handler]:
        return [
            handlers.SimpleInventoryAgentStart(
                [{"type": "diamond_axe", "quantity": 64}]
            ),
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
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    @override
    def create_server_decorators(self) -> list[Handler]:
        return []

    @override
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold  # pyright: ignore[reportOperatorIssue]

    @override
    def is_from_folder(self, folder: str) -> bool:
        return folder == "simple"

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
            handlers.ObserveFromFullStats("mine_block"),
            RayObservation(),
        ]

    @override
    def create_server_initial_conditions(self) -> list[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]
