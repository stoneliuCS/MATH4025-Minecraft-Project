import logging
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from environment.wood_environment import (
    GatherWoodEnvironment,
    LogRewardWrapper,
    StickyAttackWrapper,
    PovImageWrapper,
    RenderWrapper,
    ActionWrapper,
)
from model.environment import create_environment

logger = logging.getLogger(__name__)

TOTAL_TIMESTEPS = 500_000
CHECKPOINT_PATH = "artifacts/sac"
MODEL_PATH = "artifacts/sac_final.zip"


def run(render: bool = False):
    env_name = "GatherWood-v0"
    wood_env = GatherWoodEnvironment()
    wood_env.register()
    env = create_environment(env_name, interactive=True)

    # Wrapper stack: raw env -> reward -> sticky attack -> render -> image obs -> action mapping
    env = LogRewardWrapper(env)
    env = StickyAttackWrapper(env, sticky_ticks=15)
    if render:
        env = RenderWrapper(env)
    env = PovImageWrapper(env)
    env = ActionWrapper(env)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=100_000,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        tau=5e-3,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_PATH,
        name_prefix="sac_wood",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        log_interval=10,
    )

    model.save(MODEL_PATH)
    logger.info(f"Training complete. Model saved to {MODEL_PATH}")
    env.close()
