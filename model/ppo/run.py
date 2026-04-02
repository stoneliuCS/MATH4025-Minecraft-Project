import logging
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from environment.wood_env2 import GatherWoodEnvironment

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from model.environment import make_wood_env

logger = logging.getLogger(__name__)
from stable_baselines3.common.callbacks import BaseCallback
import os


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_norm = self.training_env

            # unwrap VecNormalize if needed
            if hasattr(vec_norm, "save"):
                path = os.path.join(
                    self.save_path, f"vecnormalize_{self.n_calls * 2 + 776000}.pkl"
                )
                vec_norm.save(path)

                if self.verbose:
                    print(f"Saved VecNormalize to {path}")

        return True


TOTAL_TIMESTEPS = 500_000
CHECKPOINT_PATH = "artifacts/ppo9"
MODEL_PATH = "artifacts/ppo_final9.zip"
VECNORM_PATH = "artifacts/vecnormalize9.pkl"
TENSORBOARD_LOG = "./tensorboard_logs9/"


def run(render: bool = False):
    env_name = "GatherWood-v0"
    wood_env = GatherWoodEnvironment()
    wood_env.register()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    n_envs = 1
    env = DummyVecEnv(
        [
            lambda: Monitor(make_wood_env(env_name, render=render, interactive=True))
            for _ in range(n_envs)
        ]
    )

    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=1000,
        batch_size=256,
        learning_rate=6e-5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=TENSORBOARD_LOG,
        device="auto",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_PATH,
        name_prefix="ppo_wood",
    )

    vecnorm_cb = SaveVecNormalizeCallback(
        save_path=CHECKPOINT_PATH,
        save_freq=10_000,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, vecnorm_cb],
        log_interval=10,
    )

    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)

    logger.info("Training complete.")
    env.close()

