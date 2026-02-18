import logging
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


MAX_EPISODE_STEPS = 2000
MAX_REWARD_THRESHOLD = 100
FRAME_SIZE = 64
CAMERA_MAX_ANGLE = 10.0
ACTION_DIM = 4  # camera_pitch, camera_yaw, forward, attack
LOG_ITEMS = ["oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log"]


class PovImageWrapper(gym.ObservationWrapper):
    """Extract 'pov' from MineRL Dict obs, resize, return as (C, H, W) uint8."""

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
        return np.transpose(img, (2, 0, 1)).astype(np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info


class LogRewardWrapper(gym.Wrapper):
    """Primary reward: +1 per log collected, -0.03 per leaf/sapling picked up."""

    LEAF_ITEMS = [
        "oak_leaves", "spruce_leaves", "birch_leaves",
        "jungle_leaves", "acacia_leaves", "dark_oak_leaves",
        "oak_sapling", "spruce_sapling", "birch_sapling",
        "jungle_sapling", "acacia_sapling", "dark_oak_sapling",
    ]

    def __init__(self, env, reward_per_log: float = 1.0, leaf_penalty: float = -0.03):
        super().__init__(env)
        self.reward_per_log = reward_per_log
        self.leaf_penalty = leaf_penalty
        self._prev_logs = 0
        self._prev_leaves = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_logs = self._get_log_count(obs)
        self._prev_leaves = self._get_leaf_count(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        cur_logs = self._get_log_count(obs)
        log_diff = cur_logs - self._prev_logs
        if log_diff > 0:
            reward += log_diff * self.reward_per_log
            logger.info(f"🪵 Collected log! total={cur_logs} (+{log_diff})")
            with open("artifacts/reward_log.txt", "a") as f:
                f.write(f"logs: {cur_logs} (+{log_diff}) reward: {reward}\n")
        self._prev_logs = cur_logs

        cur_leaves = self._get_leaf_count(obs)
        leaf_diff = cur_leaves - self._prev_leaves
        if leaf_diff > 0:
            reward += leaf_diff * self.leaf_penalty
            logger.debug(f"🍃 leaf/sapling pickup (+{leaf_diff}) penalty={leaf_diff * self.leaf_penalty:.3f}")
        self._prev_leaves = cur_leaves

        return obs, reward, done, info

    @staticmethod
    def _get_log_count(obs) -> int:
        if isinstance(obs, dict) and "inventory" in obs:
            inv = obs["inventory"]
            if isinstance(inv, dict):
                return sum(int(inv.get(item, 0)) for item in LOG_ITEMS)
        return 0

    def _get_leaf_count(self, obs) -> int:
        if isinstance(obs, dict) and "inventory" in obs:
            inv = obs["inventory"]
            if isinstance(inv, dict):
                return sum(int(inv.get(item, 0)) for item in self.LEAF_ITEMS)
        return 0


class PersistentMineWrapper(gym.Wrapper):
    """
    Once the agent triggers attack while looking at wood, hold attack
    continuously until the wood disappears from the center patch (block broken)
    or a max timeout is reached.

    Replaces StickyAttackWrapper entirely — don't use both.
    Place AFTER LogRewardWrapper, BEFORE WoodDetectionRewardWrapper.
    """

    WOOD_HSV_LOW  = np.array([15,  80,  80])
    WOOD_HSV_HIGH = np.array([25, 180, 160])
    LEAF_HSV_LOW  = np.array([35,  50,  40])
    LEAF_HSV_HIGH = np.array([85, 255, 180])

    CENTER_SIZE    = 32
    CONTEXT_SIZE   = 96
    WOOD_THRESHOLD = 0.15   # slightly lenient so drift mid-swing doesn't release early
    LEAF_THRESHOLD = 0.10
    MAX_HOLD_TICKS = 60     # safety cutoff (~3 sec at 20tps); oak takes ~30 ticks

    def __init__(self, env):
        super().__init__(env)
        self._holding_attack = False
        self._hold_ticks     = 0

    def reset(self, **kwargs):
        self._holding_attack = False
        self._hold_ticks     = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if isinstance(action, dict):
            action = self._maybe_override_attack(action)
            obs, reward, done, info = self.env.step(action)
            pov = obs.get("pov") if isinstance(obs, dict) else None
            self._update_hold_state(action, pov)
        else:
            obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

    def _is_looking_at_wood(self, pov) -> bool:
        if pov is None:
            return False
        h, w   = pov.shape[:2]
        cy, cx = h // 2, w // 2

        ctx_half = self.CONTEXT_SIZE // 2
        y0, y1 = max(cy - ctx_half, 0), min(cy + ctx_half, h)
        x0, x1 = max(cx - ctx_half, 0), min(cx + ctx_half, w)
        ctx_hsv   = cv2.cvtColor(pov[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
        leaf_mask = cv2.inRange(ctx_hsv, self.LEAF_HSV_LOW, self.LEAF_HSV_HIGH)
        has_leaves = np.count_nonzero(leaf_mask) / leaf_mask.size > self.LEAF_THRESHOLD

        half   = self.CENTER_SIZE // 2
        center = pov[cy - half:cy + half, cx - half:cx + half]
        hsv    = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
        wood_mask  = cv2.inRange(hsv, self.WOOD_HSV_LOW, self.WOOD_HSV_HIGH)
        wood_ratio = np.count_nonzero(wood_mask) / wood_mask.size

        leaf_center = cv2.inRange(hsv, self.LEAF_HSV_LOW, self.LEAF_HSV_HIGH)
        crosshair_on_leaves = np.count_nonzero(leaf_center) / leaf_center.size > 0.25

        return has_leaves and wood_ratio > self.WOOD_THRESHOLD and not crosshair_on_leaves

    def _maybe_override_attack(self, action: dict) -> dict:
        if self._holding_attack:
            action = dict(action)
            action["attack"] = 1
        return action

    def _update_hold_state(self, action: dict, pov):
        if not self._holding_attack:
            if action.get("attack", 0) == 1 and self._is_looking_at_wood(pov):
                self._holding_attack = True
                self._hold_ticks     = 1
                logger.debug("⛏ PersistentMine: started hold")
        else:
            self._hold_ticks += 1
            still_on_wood = self._is_looking_at_wood(pov)
            timed_out     = self._hold_ticks >= self.MAX_HOLD_TICKS

            if not still_on_wood:
                logger.debug(f"⛏ PersistentMine: released after {self._hold_ticks} ticks (wood gone)")
                self._holding_attack = False
                self._hold_ticks     = 0
            elif timed_out:
                logger.debug(f"⛏ PersistentMine: released after timeout ({self.MAX_HOLD_TICKS} ticks)")
                self._holding_attack = False
                self._hold_ticks     = 0


class WoodDetectionRewardWrapper(gym.Wrapper):
    """Visual reward shaping — rebalanced to make attacking clearly the best action."""

    WOOD_HSV_LOW  = np.array([15,  80,  80])
    WOOD_HSV_HIGH = np.array([25, 180, 160])

    LEAF_HSV_LOW  = np.array([35,  50,  40])
    LEAF_HSV_HIGH = np.array([85, 255, 180])

    CENTER_SIZE  = 32
    CONTEXT_SIZE = 96

    WOOD_THRESHOLD     = 0.20
    LEAF_THRESHOLD     = 0.10
    CENTER_LEAF_THRESH = 0.25

    LOOK_REWARD         =  0.01   # small — can't be farmed by spinning
    APPROACH_REWARD     =  0.02
    MINE_REWARD         =  0.30   # large — makes attacking clearly optimal
    DIG_PENALTY         = -0.10
    LEAF_ATTACK_PENALTY = -0.05
    RANDOM_ATK_PENALTY  = -0.01   # relaxed to not punish early exploration
    STEP_PENALTY        = -0.001  # tiny living cost to discourage aimless wandering

    def __init__(self, env):
        super().__init__(env)
        self._prev_wood_ratio = 0.0

    def reset(self, **kwargs):
        self._prev_wood_ratio = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward += self.STEP_PENALTY

        pov = obs["pov"] if isinstance(obs, dict) else None

        attacking    = isinstance(action, dict) and action.get("attack", 0) == 1
        moving_fwd   = isinstance(action, dict) and action.get("forward", 0) == 1
        cam          = action.get("camera", [0.0, 0.0]) if isinstance(action, dict) else [0.0, 0.0]
        cam_pitch    = float(cam[0]) if hasattr(cam, "__len__") else 0.0
        looking_down = cam_pitch > 15.0

        if pov is not None:
            h, w = pov.shape[:2]
            cy, cx = h // 2, w // 2

            ctx_half = self.CONTEXT_SIZE // 2
            y0, y1 = max(cy - ctx_half, 0), min(cy + ctx_half, h)
            x0, x1 = max(cx - ctx_half, 0), min(cx + ctx_half, w)
            ctx_hsv    = cv2.cvtColor(pov[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
            leaf_mask  = cv2.inRange(ctx_hsv, self.LEAF_HSV_LOW, self.LEAF_HSV_HIGH)
            leaf_ratio = np.count_nonzero(leaf_mask) / leaf_mask.size
            has_leaves = leaf_ratio > self.LEAF_THRESHOLD

            half   = self.CENTER_SIZE // 2
            center = pov[cy - half:cy + half, cx - half:cx + half]
            hsv    = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)

            wood_mask         = cv2.inRange(hsv, self.WOOD_HSV_LOW, self.WOOD_HSV_HIGH)
            wood_ratio        = np.count_nonzero(wood_mask) / wood_mask.size
            leaf_center_mask  = cv2.inRange(hsv, self.LEAF_HSV_LOW, self.LEAF_HSV_HIGH)
            leaf_center_ratio = np.count_nonzero(leaf_center_mask) / leaf_center_mask.size

            crosshair_on_leaves = leaf_center_ratio > self.CENTER_LEAF_THRESH
            looking_at_wood     = (
                has_leaves
                and wood_ratio > self.WOOD_THRESHOLD
                and not crosshair_on_leaves
            )

            if looking_at_wood:
                if attacking:
                    reward += self.MINE_REWARD
                    logger.info(f"✅ mining wood (wood={wood_ratio:.2f}, leaf={leaf_ratio:.2f}) +{self.MINE_REWARD}")
                else:
                    reward += self.LOOK_REWARD
                    logger.info(f"👀 looking at wood (wood={wood_ratio:.2f}, leaf={leaf_ratio:.2f}) +{self.LOOK_REWARD}")

                if moving_fwd and not looking_down and wood_ratio > self._prev_wood_ratio and wood_ratio > 0.05:
                    reward += self.APPROACH_REWARD
                    logger.info(f"🚶 approaching wood (wood={wood_ratio:.2f}, prev={self._prev_wood_ratio:.2f}) +{self.APPROACH_REWARD}")

                self._prev_wood_ratio = wood_ratio
            else:
                self._prev_wood_ratio = 0.0

                if attacking:
                    if crosshair_on_leaves:
                        reward += self.LEAF_ATTACK_PENALTY
                        logger.debug(f"🍃 attacking leaves {self.LEAF_ATTACK_PENALTY}")
                    else:
                        reward += self.RANDOM_ATK_PENALTY
                        logger.debug(f"⛏ attacking non-wood {self.RANDOM_ATK_PENALTY}")

        if attacking and looking_down:
            reward += self.DIG_PENALTY
            logger.debug(f"⬇ digging down {self.DIG_PENALTY}")

        return obs, reward, done, info


class CameraStabilityWrapper(gym.Wrapper):
    """Penalise large aimless camera movements to stop spinning behaviour.
    Place BEFORE ActionWrapper (receives raw 4-dim float action from SAC).
    """

    def __init__(self, env, spin_threshold: float = 0.5, spin_penalty: float = -0.03):
        super().__init__(env)
        self.spin_threshold = spin_threshold
        self.spin_penalty   = spin_penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # action may be a raw numpy array (float) or already a MineRL dict
        if isinstance(action, dict):
            cam = action.get("camera", [0.0, 0.0])
            # camera is in degrees; normalise by CAMERA_MAX_ANGLE to get [-1,1] scale
            c0 = float(cam[0]) / CAMERA_MAX_ANGLE
            c1 = float(cam[1]) / CAMERA_MAX_ANGLE
        else:
            c0 = float(action[0])
            c1 = float(action[1])
        cam_mag = (c0 ** 2 + c1 ** 2) ** 0.5
        if cam_mag > self.spin_threshold:
            reward += self.spin_penalty * (cam_mag - self.spin_threshold)
        return obs, reward, done, info


class RenderWrapper(gym.Wrapper):
    def step(self, action):
        self.env.render()
        return self.env.step(action)


class ActionWrapper(gym.ActionWrapper):
    """Map a 4-dim vector in [-1, 1] to a MineRL action dict.

    Layout:
        [0] camera pitch  — scaled to [-CAMERA_MAX_ANGLE, CAMERA_MAX_ANGLE]
        [1] camera yaw    — scaled to [-CAMERA_MAX_ANGLE, CAMERA_MAX_ANGLE]
        [2] forward       — > 0 => 1, else 0
        [3] attack        — > 0 => 1, else 0
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict:
        noop = self.env.action_space.noop()
        noop["camera"] = np.array([
            action[0] * CAMERA_MAX_ANGLE,
            action[1] * CAMERA_MAX_ANGLE,
        ], dtype=np.float32)
        noop["forward"] = int(action[2] > 0)
        noop["attack"]  = int(action[3] > 0)
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
        import os
        world_path = os.path.join(os.path.dirname(__file__), "worlds", "getwood.zip")
        return [
            handlers.LoadWorldAgentStart(world_path),
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
        ]

    @override
    def create_server_initial_conditions(self) -> list[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]