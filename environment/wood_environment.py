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

# ── Birch bark HSV range ───────────────────────────────────────────────
# Birch bark is off-white / cream with dark stripes.  With gamma=2.0 the
# bark gets washed out to near-white, so we need very permissive ranges:
# S can drop to near 0 and V can hit 255.  We accept any warm-ish hue
# with low saturation and high value.
BIRCH_WOOD_HSV_LOW  = np.array([0,    0,  130])
BIRCH_WOOD_HSV_HIGH = np.array([35, 100,  255])

# Dark horizontal stripe pixels — near-black with minimal saturation.
BIRCH_STRIPE_HSV_LOW  = np.array([0,   0,   20])
BIRCH_STRIPE_HSV_HIGH = np.array([180, 50,  130])

# ── Birch leaf HSV range ──────────────────────────────────────────────
# Birch leaves #80a755 → OpenCV HSV ≈ H44, S125, V167.
# With gamma=2.0 values shift brighter.  Keep range generous.
BIRCH_LEAF_HSV_LOW  = np.array([30,  50,  80])
BIRCH_LEAF_HSV_HIGH = np.array([55, 220, 230])


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


def _birch_wood_mask(hsv_patch: np.ndarray) -> np.ndarray:
    """Return a combined binary mask that matches birch bark (pale body + dark stripes)."""
    mask_body   = cv2.inRange(hsv_patch, BIRCH_WOOD_HSV_LOW,   BIRCH_WOOD_HSV_HIGH)
    mask_stripe = cv2.inRange(hsv_patch, BIRCH_STRIPE_HSV_LOW, BIRCH_STRIPE_HSV_HIGH)
    return cv2.bitwise_or(mask_body, mask_stripe)


def _birch_leaf_mask(hsv_patch: np.ndarray) -> np.ndarray:
    """Return a binary mask for birch leaves."""
    return cv2.inRange(hsv_patch, BIRCH_LEAF_HSV_LOW, BIRCH_LEAF_HSV_HIGH)


def _has_leaves_above(pov: np.ndarray, context_size: int, leaf_threshold: float) -> bool:
    """Check for birch leaves in the UPPER portion of the context patch only.

    This avoids false positives from grass on the ground, which shares
    a similar hue with birch leaves.  Real leaves from a tree canopy
    appear in the upper half of the viewport when the agent is near a trunk.
    """
    h, w = pov.shape[:2]
    cy, cx = h // 2, w // 2
    ctx_half = context_size // 2

    y0 = max(cy - ctx_half, 0)
    y1 = cy  # only the upper half of the context (above crosshair)
    x0 = max(cx - ctx_half, 0)
    x1 = min(cx + ctx_half, w)

    if y1 <= y0 or x1 <= x0:
        return False

    upper_patch = pov[y0:y1, x0:x1]
    hsv = cv2.cvtColor(upper_patch, cv2.COLOR_RGB2HSV)
    leaf_mask = _birch_leaf_mask(hsv)
    leaf_ratio = np.count_nonzero(leaf_mask) / leaf_mask.size
    return leaf_ratio > leaf_threshold


class PersistentMineWrapper(gym.Wrapper):
    """
    Once the agent triggers attack while looking at wood, hold attack
    continuously until the wood disappears from the center patch (block broken)
    or a max timeout is reached.

    Key fix: once holding, we require several consecutive frames of
    "no wood" before releasing — this prevents flickering from slight
    camera drift mid-swing from causing premature release.
    """

    CENTER_SIZE    = 32
    CONTEXT_SIZE   = 96
    WOOD_THRESHOLD = 0.12   # slightly more lenient to stay locked on
    LEAF_THRESHOLD = 0.08
    MAX_HOLD_TICKS = 80     # birch takes ~30 ticks bare-hand; give extra margin
    RELEASE_GRACE  = 5      # must see "no wood" for this many consecutive ticks to release

    def __init__(self, env):
        super().__init__(env)
        self._holding_attack = False
        self._hold_ticks     = 0
        self._no_wood_streak = 0

    def reset(self, **kwargs):
        self._holding_attack = False
        self._hold_ticks     = 0
        self._no_wood_streak = 0
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

        # Center patch — check for birch wood
        half   = self.CENTER_SIZE // 2
        center = pov[cy - half:cy + half, cx - half:cx + half]
        hsv    = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
        wood_mask  = _birch_wood_mask(hsv)
        wood_ratio = np.count_nonzero(wood_mask) / wood_mask.size

        leaf_center = _birch_leaf_mask(hsv)
        crosshair_on_leaves = np.count_nonzero(leaf_center) / leaf_center.size > 0.25

        if crosshair_on_leaves:
            return False

        # If wood ratio is very high, trust it even without visible leaves
        # (agent may be face-to-face with trunk, canopy off-screen)
        if wood_ratio > 0.40:
            return True

        # Otherwise require leaves above to confirm it's a tree
        has_leaves = _has_leaves_above(pov, self.CONTEXT_SIZE, self.LEAF_THRESHOLD)
        return has_leaves and wood_ratio > self.WOOD_THRESHOLD

    def _maybe_override_attack(self, action: dict) -> dict:
        if self._holding_attack:
            action = dict(action)
            action["attack"] = 1
            # Freeze camera and movement while mining — prevent drifting off target
            action["camera"] = np.array([0.0, 0.0], dtype=np.float32)
            action["forward"] = 0
            action["back"] = 0
            action["left"] = 0
            action["right"] = 0
            action["jump"] = 0
        return action

    def _update_hold_state(self, action: dict, pov):
        if not self._holding_attack:
            if action.get("attack", 0) == 1 and self._is_looking_at_wood(pov):
                self._holding_attack = True
                self._hold_ticks     = 1
                self._no_wood_streak = 0
                logger.debug("⛏ PersistentMine: started hold")
        else:
            self._hold_ticks += 1
            still_on_wood = self._is_looking_at_wood(pov)
            timed_out     = self._hold_ticks >= self.MAX_HOLD_TICKS

            if still_on_wood:
                self._no_wood_streak = 0
            else:
                self._no_wood_streak += 1

            if self._no_wood_streak >= self.RELEASE_GRACE:
                logger.debug(f"⛏ PersistentMine: released after {self._hold_ticks} ticks "
                             f"(wood gone for {self.RELEASE_GRACE} frames)")
                self._holding_attack = False
                self._hold_ticks     = 0
                self._no_wood_streak = 0
            elif timed_out:
                logger.debug(f"⛏ PersistentMine: released after timeout ({self.MAX_HOLD_TICKS} ticks)")
                self._holding_attack = False
                self._hold_ticks     = 0
                self._no_wood_streak = 0


class WoodDetectionRewardWrapper(gym.Wrapper):
    """Visual reward shaping — tuned for birch bark detection.

    Uses upper-viewport leaf check to avoid confusing grass with leaves.
    """

    CENTER_SIZE  = 32
    CONTEXT_SIZE = 96

    WOOD_THRESHOLD     = 0.20
    LEAF_THRESHOLD     = 0.08
    CENTER_LEAF_THRESH = 0.25

    LOOK_REWARD         =  0.01
    MINE_REWARD         =  0.30
    DIG_PENALTY         = -0.10
    LEAF_ATTACK_PENALTY = -0.05
    RANDOM_ATK_PENALTY  = -0.01
    STEP_PENALTY        = -0.001

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

            # Check for leaves ABOVE the crosshair only (not grass)
            has_leaves = _has_leaves_above(pov, self.CONTEXT_SIZE, self.LEAF_THRESHOLD)

            # Also compute full context leaf ratio for logging
            ctx_half = self.CONTEXT_SIZE // 2
            y0, y1 = max(cy - ctx_half, 0), min(cy + ctx_half, h)
            x0, x1 = max(cx - ctx_half, 0), min(cx + ctx_half, w)
            ctx_hsv    = cv2.cvtColor(pov[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
            leaf_mask  = _birch_leaf_mask(ctx_hsv)
            leaf_ratio = np.count_nonzero(leaf_mask) / leaf_mask.size

            # Center patch — birch wood
            half   = self.CENTER_SIZE // 2
            center = pov[cy - half:cy + half, cx - half:cx + half]
            hsv    = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)

            wood_mask         = _birch_wood_mask(hsv)
            wood_ratio        = np.count_nonzero(wood_mask) / wood_mask.size
            leaf_center_mask  = _birch_leaf_mask(hsv)
            leaf_center_ratio = np.count_nonzero(leaf_center_mask) / leaf_center_mask.size

            crosshair_on_leaves = leaf_center_ratio > self.CENTER_LEAF_THRESH

            # If wood ratio is very high, trust it without leaf confirmation
            # (agent may be face-to-face with trunk, canopy off-screen)
            if crosshair_on_leaves:
                looking_at_wood = False
            elif wood_ratio > 0.40:
                looking_at_wood = True
            else:
                looking_at_wood = has_leaves and wood_ratio > self.WOOD_THRESHOLD

            if looking_at_wood:
                if attacking:
                    reward += self.MINE_REWARD
                    logger.info(f"✅ mining birch (wood={wood_ratio:.2f}, leaf={leaf_ratio:.2f}) +{self.MINE_REWARD}")
                else:
                    reward += self.LOOK_REWARD
                    logger.info(f"👀 looking at birch (wood={wood_ratio:.2f}, leaf={leaf_ratio:.2f}) +{self.LOOK_REWARD}")

                self._prev_wood_ratio = wood_ratio
            else:
                self._prev_wood_ratio = 0.0

                if attacking:
                    if crosshair_on_leaves and has_leaves:
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
    """Penalise large aimless camera movements to stop spinning behaviour."""

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


class RenderWrapper(gym.Wrapper):
    def step(self, action):
        self.env.render()
        return self.env.step(action)


class ActionWrapper(gym.ActionWrapper):
    """Map a 4-dim vector in [-1, 1] to a MineRL action dict."""

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
        world_path = os.path.join(os.path.dirname(__file__), "worlds", "birch_trees.zip")
        return [
            handlers.LoadWorldAgentStart(world_path),
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