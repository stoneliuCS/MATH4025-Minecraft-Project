import logging
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from environment.wood_environment import (
    GatherWoodEnvironment,
    LogRewardWrapper,
    PersistentMineWrapper,
    WoodDetectionRewardWrapper,
    CameraStabilityWrapper,
    PovImageWrapper,
    RenderWrapper,
    ActionWrapper,
)
from model.environment import create_environment

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "artifacts/sac"
MODEL_PATH      = "artifacts/sac_final.zip"


class EpisodeLogCallback(BaseCallback):
    """Log per-episode stats and attack% to help diagnose stuck policies."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._attack_ticks = 0
        self._total_ticks  = 0

    def _on_step(self) -> bool:
        action = self.locals.get("actions")
        if action is not None and hasattr(action, "__len__"):
            if action[0][3] > 0:
                self._attack_ticks += 1
        self._total_ticks += 1

        if self._total_ticks % 500 == 0:
            attack_pct = 100.0 * self._attack_ticks / max(self._total_ticks, 1)
            logger.info(f"[step {self._total_ticks}] attack%={attack_pct:.1f}%")

        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                attack_pct = 100.0 * self._attack_ticks / max(self._total_ticks, 1)
                logger.info(
                    f"Episode finished: reward={ep_reward:.2f}, "
                    f"length={ep_length}, attack%={attack_pct:.1f}%"
                )
                self._attack_ticks = 0
                self._total_ticks  = 0
        return True


def run(render: bool = True, small_training: bool = False):
    env_name = "GatherWood-v0"
    GatherWoodEnvironment().register()

    env = create_environment(env_name, interactive=render)

    # Wrapper stack — order matters
    env = LogRewardWrapper(env)                        # true log reward
    env = PersistentMineWrapper(env)                   # hold attack until block breaks
    env = WoodDetectionRewardWrapper(env)              # visual shaping
    env = CameraStabilityWrapper(env,                  # anti-spin
              spin_threshold=0.5, spin_penalty=-0.03)
    if render:
        env = RenderWrapper(env)
    env = PovImageWrapper(env)    # dict → (C,H,W) uint8
    env = ActionWrapper(env)      # 4-dim float → MineRL dict

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if small_training:
        timesteps       = 10_000
        buffer_size     = 30_000
        learning_starts = 2_000
        save_freq       = 2_000
    else:
        timesteps       = 50_000
        buffer_size     = 100_000
        learning_starts = 5_000
        save_freq       = 5_000

    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=buffer_size,
        batch_size=256,
        learning_rate=1e-4,
        gamma=0.99,
        tau=5e-3,
        train_freq=8,
        gradient_steps=4,
        learning_starts=learning_starts,
        ent_coef="auto",
        target_entropy=-2.0,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=CHECKPOINT_PATH,
        name_prefix="sac_wood",
    )
    episode_cb = EpisodeLogCallback()

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, episode_cb],
        log_interval=10,
    )

    model.save(MODEL_PATH)
    logger.info(f"Training complete. Model saved to {MODEL_PATH}")
    env.close()


if __name__ == "__main__":
    run(render=True, small_training=False)