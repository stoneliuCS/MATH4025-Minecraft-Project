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


MAX_EPISODE_STEPS = 500
MAX_REWARD_THRESHOLD = 100
FRAME_SIZE = 64
CAMERA_MAX_ANGLE = 5.0
ACTION_DIM = 5  # camera_pitch, camera_yaw, forward, attack, jump
LOG_ITEMS = ["oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log"]

# ── Birch bark HSV range ───────────────────────────────────────────────
BIRCH_WOOD_HSV_LOW  = np.array([0,    0,  130])
BIRCH_WOOD_HSV_HIGH = np.array([35, 100,  255])
BIRCH_STRIPE_HSV_LOW  = np.array([0,   0,   20])
BIRCH_STRIPE_HSV_HIGH = np.array([180, 50,  130])

# ── Birch leaf HSV range ──────────────────────────────────────────────
BIRCH_LEAF_HSV_LOW  = np.array([30,  50,  80])
BIRCH_LEAF_HSV_HIGH = np.array([55, 220, 230])

MIN_VERTICAL_FILL = 0.40

# ── Streak bonus config ───────────────────────────────────────────────
STREAK_START        = 5    # ticks before any bonus kicks in
STREAK_BONUS_PER_TICK = 0.04  # reward per tick after STREAK_START
STREAK_MAX_BONUS    = 0.3  # cap per tick so it doesn't dominate


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
    """Primary reward: +4 per log collected, -0.03 per leaf/sapling picked up.
    Also writes logs_collected into info so downstream wrappers can read it.
    """

    LEAF_ITEMS = [
        "oak_leaves", "spruce_leaves", "birch_leaves",
        "jungle_leaves", "acacia_leaves", "dark_oak_leaves",
        "oak_sapling", "spruce_sapling", "birch_sapling",
        "jungle_sapling", "acacia_sapling", "dark_oak_sapling",
    ]

    def __init__(self, env, reward_per_log: float = 5.0, leaf_penalty: float = -0.03):
        super().__init__(env)
        self.reward_per_log = reward_per_log
        self.leaf_penalty = leaf_penalty
        self._prev_logs = 0
        self._prev_leaves = 0
        self._total_logs = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_logs = self._get_log_count(obs)
        self._prev_leaves = self._get_leaf_count(obs)
        self._total_logs = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        cur_logs = self._get_log_count(obs)
        log_diff = cur_logs - self._prev_logs
        if log_diff > 0:
            reward += log_diff * self.reward_per_log
            self._total_logs += log_diff
            logger.info(f"🪵 Collected log! total={cur_logs} (+{log_diff})")
            with open("artifacts/reward_log.txt", "a") as f:
                f.write(f"logs: {cur_logs} (+{log_diff}) reward: {reward}\n")
        self._prev_logs = cur_logs

        # ✅ Write into info so WoodDetectionRewardWrapper can read it
        info["logs_collected"] = self._total_logs

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


def _birch_wood_mask(hsv_patch: np.ndarray) -> np.ndarray:
    mask_body   = cv2.inRange(hsv_patch, BIRCH_WOOD_HSV_LOW,   BIRCH_WOOD_HSV_HIGH)
    mask_stripe = cv2.inRange(hsv_patch, BIRCH_STRIPE_HSV_LOW, BIRCH_STRIPE_HSV_HIGH)
    return cv2.bitwise_or(mask_body, mask_stripe)


def _birch_leaf_mask(hsv_patch: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv_patch, BIRCH_LEAF_HSV_LOW, BIRCH_LEAF_HSV_HIGH)


def _has_leaves_above(pov: np.ndarray, context_size: int, leaf_threshold: float) -> bool:
    h, w = pov.shape[:2]
    cy, cx = h // 2, w // 2
    ctx_half = context_size // 2
    y0 = max(cy - ctx_half, 0)
    y1 = cy
    x0 = max(cx - ctx_half, 0)
    x1 = min(cx + ctx_half, w)
    if y1 <= y0 or x1 <= x0:
        return False
    upper_patch = pov[y0:y1, x0:x1]
    hsv = cv2.cvtColor(upper_patch, cv2.COLOR_RGB2HSV)
    leaf_mask = _birch_leaf_mask(hsv)
    return np.count_nonzero(leaf_mask) / leaf_mask.size > leaf_threshold


def _is_close_trunk(pov: np.ndarray, center_size: int) -> bool:
    """Vertical column check — only triggers on nearby trunks."""
    h, w = pov.shape[:2]
    cy, cx = h // 2, w // 2
    half = center_size // 2
    center = pov[cy - half:cy + half, cx - half:cx + half]
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    wood_mask = _birch_wood_mask(hsv)
    patch_h = center.shape[0]
    if patch_h == 0:
        return False
    cols_to_check = [center_size // 4, center_size // 2, 3 * center_size // 4]
    best_run = 0
    for col in cols_to_check:
        if col >= wood_mask.shape[1]:
            continue
        column = wood_mask[:, col]
        run = 0
        max_run = 0
        for pixel in column:
            if pixel > 0:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        best_run = max(best_run, max_run)
    return best_run / patch_h >= MIN_VERTICAL_FILL


def _detect_close_wood(pov: np.ndarray, center_size: int) -> bool:
    """Simple, robust detection: just check if enough wood-colored pixels exist."""
    if pov is None:
        return False

    h, w = pov.shape[:2]
    cy, cx = h // 2, w // 2
    half = center_size // 2
    center = pov[cy - half:cy + half, cx - half:cx + half]

    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    wood_mask = cv2.inRange(hsv, BIRCH_WOOD_HSV_LOW, BIRCH_WOOD_HSV_HIGH)

    return np.mean(wood_mask > 0) > 0.1


class WoodDetectionRewardWrapper(gym.Wrapper):
    CENTER_SIZE = 28

    # # Small incentive to face wood — not enough to farm on its own
    # LOOK_REWARD         = 0.02

    # # Small baseline reward every tick the agent attacks wood.
    # # Streak bonus stacks on top after STREAK_START consecutive ticks.
    # MINE_REWARD         = 0.05
    # STEADY_AIM_BONUS    = 0.05   # extra per tick when camera is steady during streak

    # # Collection reward lives in LogRewardWrapper (4.0). This is the info-key
    # # top-up for any logs WoodDetectionRewardWrapper notices via info — kept
    # # at 0 to avoid double-counting now that LogRewardWrapper owns that signal.
    # LOG_COLLECT_REWARD  = 0.0
    LOOK_REWARD = 0.0
    MINE_REWARD = 0.02
    STREAK_START = 8
    STREAK_BONUS_PER_TICK = 0.06
    STREAK_MAX_BONUS = 0.3
    STEADY_AIM_BONUS = 0.05

    def __init__(self, env):
        super().__init__(env)
        self._consecutive_mine = 0
        self._last_logs_collected = 0

    def reset(self, **kwargs):
        self._consecutive_mine = 0
        self._last_logs_collected = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        pov = obs.get("pov") if isinstance(obs, dict) else None
        attacking = isinstance(action, dict) and action.get("attack", 0) == 1
        cam = action.get("camera", [0.0, 0.0]) if isinstance(action, dict) else [0.0, 0.0]
        cam_mag = (float(cam[0]) ** 2 + float(cam[1]) ** 2) ** 0.5

        if pov is not None:
            looking_at_wood = _detect_close_wood(pov, self.CENTER_SIZE)

            if attacking and self._consecutive_mine > 20 and not looking_at_wood:
                reward -= 0.1

            if looking_at_wood:
                if attacking:
                    self._consecutive_mine += 1

                    # Flat reward every attack-on-wood tick.
                    reward += self.MINE_REWARD

                    # Streak bonus stacks on top after STREAK_START ticks,
                    # capped so it can never dwarf actual log collection (4.0).
                    if self._consecutive_mine > STREAK_START:
                        bonus = min(
                            STREAK_BONUS_PER_TICK * (self._consecutive_mine - STREAK_START),
                            STREAK_MAX_BONUS,
                        )
                        reward += bonus
                        if cam_mag < 1.0:
                            reward += self.STEADY_AIM_BONUS
                else:
                    # Reward for facing wood without attacking — tiny,
                    # just enough to help the agent orient toward trees.
                    reward += self.LOOK_REWARD
                    self._consecutive_mine = 0
            else:
                self._consecutive_mine = 0

        # ✅ info["logs_collected"] is now set upstream by LogRewardWrapper.
        # WoodDetectionRewardWrapper no longer adds its own collection reward
        # to avoid double-counting. We just track the key for info purposes.
        current_logs = info.get("logs_collected", 0)
        self._last_logs_collected = current_logs

        info["mining_ticks"] = self._consecutive_mine
        return obs, reward, done, info


class CameraStabilityWrapper(gym.Wrapper):
    """Penalise large aimless camera movements to stop spinning."""

    def __init__(self, env, spin_threshold: float = 0.5, spin_penalty: float = -0.03):
        super().__init__(env)
        self.spin_threshold = spin_threshold
        self.spin_penalty   = spin_penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(action, dict):
            cam = action.get("camera", [0.0, 0.0])
            c0 = float(cam[0]) / CAMERA_MAX_ANGLE
            c1 = float(cam[1]) / CAMERA_MAX_ANGLE
        else:
            c0 = float(action[0])
            c1 = float(action[1])
        cam_mag = (c0 ** 2 + c1 ** 2) ** 0.5
        if cam_mag > self.spin_threshold:
            reward += self.spin_penalty * (cam_mag - self.spin_threshold)
        return obs, reward, done, info


class PersistentMineWrapper(gym.Wrapper):
    CENTER_SIZE = 28
    HOLD_TICKS = 60
    RELEASE_GRACE = 4

    def __init__(self, env):
        super().__init__(env)
        self._holding = False
        self._ticks = 0
        self._no_wood = 0

    def reset(self, **kwargs):
        self._holding = False
        self._ticks = 0
        self._no_wood = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if isinstance(action, dict) and self._holding:
            action["attack"] = 1
            action["camera"] = np.array([0.0, 0.0], dtype=np.float32)
            action["forward"] = 0
            action["jump"] = 0

        obs, reward, done, info = self.env.step(action)
        pov = obs.get("pov") if isinstance(obs, dict) else None

        if isinstance(action, dict):
            if not self._holding:
                if action.get("attack", 0) == 1 and _detect_close_wood(pov, self.CENTER_SIZE):
                    self._holding = True
                    self._ticks = 1
                    self._no_wood = 0
            else:
                self._ticks += 1
                still_wood = _detect_close_wood(pov, self.CENTER_SIZE)

                if still_wood:
                    self._no_wood = 0
                else:
                    self._no_wood += 1

                if self._no_wood >= self.RELEASE_GRACE or self._ticks >= self.HOLD_TICKS:
                    self._holding = False
                    self._ticks = 0
                    self._no_wood = 0

        return obs, reward, done, info

class RenderWrapper(gym.Wrapper):
    def step(self, action):
        self.env.render()
        return self.env.step(action)

class ResetRetryWrapper(gym.Wrapper):
    """Retry reset() on TimeoutError/OSError from MineRL's Malmo socket.
 
    The Minecraft Java process occasionally takes longer than the default
    socket timeout to load a world zip, causing a TimeoutError mid-reset.
    This wrapper catches that, waits for the underlying process to fully
    die, and retries up to `max_retries` times before re-raising.
    """
 
    def __init__(self, env, max_retries: int = 5, retry_delay: float = 5.0):
        super().__init__(env)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
 
    def reset(self, **kwargs):
        import time
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.env.reset(**kwargs)
            except (TimeoutError, OSError, ConnectionResetError) as e:
                last_exc = e
                logger.warning(
                    f"⚠️  Reset attempt {attempt}/{self.max_retries} failed "
                    f"({type(e).__name__}: {e}). Retrying in {self.retry_delay}s…"
                )
                time.sleep(self.retry_delay)
        raise RuntimeError(
            f"Env reset failed after {self.max_retries} attempts."
        ) from last_exc

class ActionWrapper(gym.ActionWrapper):
    """Map a 5-dim vector in [-1, 1] to a MineRL action dict.

    [0] camera pitch, [1] camera yaw, [2] forward, [3] attack, [4] jump
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
        noop["jump"]    = int(action[4] > 0)
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
        world_path = os.path.join(os.path.dirname(__file__), "worlds", "birch_trees.zip")
        return [
            handlers.LoadWorldAgentStart(world_path),
            handlers.SimpleInventoryAgentStart([
                {"type": "diamond_axe", "quantity": 1},
            ]),
            handlers.GammaSetting(2.0),
            handlers.FOVSetting(70.0),
            handlers.FakeCursorSize(16),
            handlers.GuiScale(1),
            handlers.PreferredSpawnBiome("birch_forest"),
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