import logging
import os
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.wood_env2 import (
    ActionWrapper,
    GatherWoodEnvironment,
    MineBlockRewardWrapper,
    PovImageWrapper,
    RenderWrapper,
    StickyAttackWrapper,
)
from environment.wrappers import RobustResetWrapper
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from model.sac.callbacks import RewardPlotCallback
from model.sac.replay_buffer import NStepReplayBuffer
from model.environment import create_environment

logger = logging.getLogger(__name__)

TOTAL_TIMESTEPS = 100_000
CHECKPOINT_PATH = "artifacts/sac"
MODEL_PATH = "artifacts/sac_final.zip"


def run(
    render: bool = False,
    checkpoint: str | None = None,
    pretrained: str | None = None,
    timesteps: int = TOTAL_TIMESTEPS,
    checkpoint_out: str = CHECKPOINT_PATH,
):
    if checkpoint:
        checkpoint_out = os.path.dirname(os.path.abspath(checkpoint))
    env_name = "GatherWood-v0"

    wood_env = GatherWoodEnvironment()
    wood_env.register()

    env = create_environment(env_name, interactive=True)
    env = RobustResetWrapper(env, env_name=env_name)
    env = MineBlockRewardWrapper(env)
    env = StickyAttackWrapper(env, sticky_ticks=5)
    if render:
        env = RenderWrapper(env)
    env = PovImageWrapper(env)
    env = ActionWrapper(env)
    env = GymV21CompatibilityV0(
        env_id=env_name, env=env
    )  # convert old gym → gymnasium at the boundary for SB3/Monitor
    env = Monitor(env, filename=os.path.join(checkpoint_out, f"monitor_{int(time.time())}"))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        model = SAC.load(checkpoint, env=env)
        model.learning_starts = model.num_timesteps + 500  # collect fresh steps before training on empty buffer
    elif pretrained:
        logger.info(f"Loading pretrained BC weights: {pretrained}")
        model = SAC.load(pretrained, env=env)
        model.learning_starts = 0  # network already initialised — skip random warmup
    else:
        model = SAC(
            "CnnPolicy",
            env,
            verbose=1,
            buffer_size=500_000,
            batch_size=512,
            learning_rate=3e-4,
            gamma=0.99,
            tau=5e-3,
            train_freq=4,
            gradient_steps=8,
            learning_starts=500,
            replay_buffer_class=NStepReplayBuffer,
            replay_buffer_kwargs={"n_steps": 10, "gamma": 0.99},
        )

    os.makedirs(checkpoint_out, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_out,
        name_prefix="sac_wood",
    )
    reward_cb = RewardPlotCallback(
        output_path=os.path.join(checkpoint_out, "reward_plot.png")
    )

    callbacks = [checkpoint_cb, reward_cb]

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList(callbacks),
        log_interval=10,
        reset_num_timesteps=checkpoint is None,
    )

    model.save(MODEL_PATH)
    logger.info(f"Training complete. Model saved to {MODEL_PATH}")
    env.close()
