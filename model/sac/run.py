import logging
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from model.sac.callbacks import RewardPlotCallback
from model.sac.replay_buffer import NStepReplayBuffer
from model.sac.bc_callback import BCRegularizationCallback

from environment.wood_environment import (
    GatherWoodEnvironment,
    LogRewardWrapper,
    StickyAttackWrapper,
    WoodDetectionRewardWrapper,
    PovImageWrapper,
    RenderWrapper,
    ActionWrapper,
)
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
    env_name = "GatherWood-v0"
    wood_env = GatherWoodEnvironment()
    wood_env.register()
    env = create_environment(env_name, interactive=True)

    # Wrapper stack: raw env -> reward -> sticky attack -> wood detection -> render -> image obs -> action mapping
    env = LogRewardWrapper(env, log_dir=checkpoint_out)
    env = StickyAttackWrapper(env, sticky_ticks=15)
    env = WoodDetectionRewardWrapper(env)
    if render:
        env = RenderWrapper(env)
    env = PovImageWrapper(env)
    env = ActionWrapper(env)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        model = SAC.load(checkpoint, env=env)
    elif pretrained:
        logger.info(f"Loading pretrained BC weights: {pretrained}")
        model = SAC.load(pretrained, env=env)
        model.learning_starts = 0  # network already initialised — skip random warmup
    else:
        model = SAC(
            "CnnPolicy",
            env,
            verbose=1,
            buffer_size=100_000,
            batch_size=512,
            learning_rate=1e-5,
            gamma=0.99,
            tau=5e-3,
            train_freq=4,
            gradient_steps=8,
            learning_starts=500,
            replay_buffer_class=NStepReplayBuffer,
            replay_buffer_kwargs={"n_steps": 50, "gamma": 0.99},
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
