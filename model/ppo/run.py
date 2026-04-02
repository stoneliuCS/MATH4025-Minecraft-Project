import logging
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from environment.wood_env2 import GatherWoodEnvironment
from model.callbacks import RewardPlotCallback
from model.environment import make_wood_env

logger = logging.getLogger(__name__)

TOTAL_TIMESTEPS = 250_000
CHECKPOINT_PATH = "artifacts/ppo"
MODEL_PATH = "artifacts/ppo_final.zip"


def run(
    render: bool = False,
    checkpoint: str | None = None,
    timesteps: int = TOTAL_TIMESTEPS,
    checkpoint_out: str = CHECKPOINT_PATH,
):
    if checkpoint:
        checkpoint_out = os.path.dirname(os.path.abspath(checkpoint))

    env_name = "GatherWood-v0"
    wood_env = GatherWoodEnvironment()
    wood_env.register()

    os.makedirs(checkpoint_out, exist_ok=True)

    monitor_path = os.path.join(checkpoint_out, f"monitor_{int(time.time())}")

    def _make():
        env = make_wood_env(env_name, render=render, interactive=True)
        return Monitor(env, filename=monitor_path)

    env = DummyVecEnv([_make])
    env = VecFrameStack(env, n_stack=4)

    vecnorm_path = os.path.join(checkpoint_out, "vecnormalize.pkl")

    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = False
            logger.info(f"Loaded VecNormalize stats from {vecnorm_path}")
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=False)
        model = PPO.load(checkpoint, env=env, device="auto")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=1e-4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            tensorboard_log=os.path.join(checkpoint_out, "tensorboard"),
            device="auto",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=checkpoint_out,
        name_prefix="ppo_wood",
    )
    reward_cb = RewardPlotCallback(
        output_path=os.path.join(checkpoint_out, "reward_plot.png"),
        title="PPO Training — Episode Rewards",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList([checkpoint_cb, reward_cb]),
        log_interval=10,
        reset_num_timesteps=checkpoint is None,
    )

    model.save(os.path.join(checkpoint_out, "ppo_final"))
    env.save(vecnorm_path)
    logger.info(f"Training complete. Model saved to {checkpoint_out}/ppo_final.zip")
    env.close()
