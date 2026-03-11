import logging
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from environment.wood_env2 import (
    GatherWoodEnvironment,
    MineBlockRewardWrapper,
    PovImageWrapper,
    RenderWrapper,
    ActionWrapper,
    StickyAttackWrapper,
    SafeMineRLWrapper
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from model.environment import create_environment

logger = logging.getLogger(__name__)

TOTAL_TIMESTEPS = 250_000
CHECKPOINT_PATH = "artifacts/ppo4"
MODEL_PATH = "artifacts/ppo_final4.zip"
VECNORM_PATH = "artifacts/vecnormalize4.pkl"
TENSORBOARD_LOG = "./tensorboard_logs4/"


# def run(render: bool = False):
#     env_name = "GatherWood-v0"
#     wood_env = GatherWoodEnvironment()
#     wood_env.register()
#     env = create_environment(env_name, interactive=True)

#     # Wrapper stack: raw env -> reward -> sticky attack -> wood detection -> render -> image obs -> action mapping
#     env = MineBlockRewardWrapper(env)
#     # env = LookAtWoodRewardWrapper(env)  # ✅ new
#     env = StickyAttackWrapper(env, sticky_ticks=15)
#     # env = WoodDetectionRewardWrapper(env)
#     if render:
#         env = RenderWrapper(env)
#     env = PovImageWrapper(env)
#     env = ActionWrapper(env)
#     env = Monitor(env)


#     env = DummyVecEnv([lambda: env])

#     env = VecFrameStack(env, n_stack=4)

#     env = VecNormalize(env, norm_obs=False, norm_reward=True)


#     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

#     model = PPO(
#         "CnnPolicy",
#         env,
#         verbose=1,
#         n_steps=1024,
#         batch_size=256,
#         learning_rate=3e-4,
#         gamma=0.995,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.05,
#         vf_coef=0.5,
#         tensorboard_log="./tensorboard_logs/",
#         device="auto",
#     )

#     checkpoint_cb = CheckpointCallback(
#         save_freq=10_000,
#         save_path=CHECKPOINT_PATH,
#         name_prefix="ppo_wood",
#     )

#     model = model.learn(
#         total_timesteps=TOTAL_TIMESTEPS,
#         callback=checkpoint_cb,
#         log_interval=10,
#     )

#     model.save(MODEL_PATH)
#     logger.info(f"Training complete. Model saved to {MODEL_PATH}")
#     env.close()

def make_env(render=False):
    env_name = "GatherWood-v0"
    wood_env = GatherWoodEnvironment()
    wood_env.register()

    env = create_environment(env_name, interactive=True)

    # env = SafeMineRLWrapper(env)  # ✅ new

    env = MineBlockRewardWrapper(env)
    env = StickyAttackWrapper(env, sticky_ticks=15)

    if render:
        env = RenderWrapper(env)

    env = PovImageWrapper(env)
    env = ActionWrapper(env)

    # Required for reward logging
    env = Monitor(env)

    return env


def run(render: bool = False):
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)


    n_envs = 1
    env = DummyVecEnv([lambda: make_env(render=False) for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
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
        tensorboard_log=TENSORBOARD_LOG,
        device="auto",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=CHECKPOINT_PATH,
        name_prefix="ppo_wood",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        log_interval=10,
    )

    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)

    logger.info("Training complete.")
    env.close()

# def run(render: bool = False, additional_timesteps: int = 280_000):
#     n_envs = 2

#     # Recreate env
#     env = DummyVecEnv([lambda: make_env(render=False) for _ in range(n_envs)])
#     env = VecFrameStack(env, n_stack=4)

#     env = VecNormalize.load(VECNORM_PATH, env)
#     env.training = True
#     env.norm_reward = True

#     model = PPO.load(MODEL_PATH, env=env, device="auto")

#     # Continue training
#     model.learn(
#         total_timesteps=additional_timesteps,
#         reset_num_timesteps=False,   # VERY IMPORTANT
#         log_interval=10,
#     )

#     # Save again
#     model.save(MODEL_PATH)
#     env.save(VECNORM_PATH)

#     print("Resumed training complete.")
#     env.close()

# def run(render: bool = False):
#     # Register once
#     wood_env = GatherWoodEnvironment()
#     wood_env.register()

#     # Single environment for evaluation
#     env = DummyVecEnv([lambda: make_env(render=True)])
#     env = VecFrameStack(env, n_stack=4)

#     # Load normalization stats
#     env = VecNormalize.load(VECNORM_PATH, env)
#     env.training = False
#     env.norm_reward = False

#     # Load trained model
#     model = PPO.load(MODEL_PATH, env=env)

#     obs = env.reset()

#     while True:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)

#         if done:
#             obs = env.reset()