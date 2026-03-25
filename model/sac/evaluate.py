import logging
from stable_baselines3 import SAC
from environment.wood_environment import (
    GatherWoodEnvironment,
    LogRewardWrapper,
    PersistentMineWrapper,
    ResetRetryWrapper,
    WoodDetectionRewardWrapper,
    CameraStabilityWrapper,
    PovImageWrapper,
    RenderWrapper,
    ActionWrapper,
)
from model.environment import create_environment

logger = logging.getLogger(__name__)
# SB3 appends .zip automatically — do NOT include the extension here
MODEL_PATH = "artifacts/sac/sac_wood_30000_steps"


def evaluate(n_episodes: int = 5, render: bool = True):
    env_name = "GatherWood-v0"
    GatherWoodEnvironment().register()
    env = create_environment(env_name, interactive=render)

    # Wrapper stack — must match run.py exactly
    env = LogRewardWrapper(env)
    env = PersistentMineWrapper(env)     # 🔥 CRITICAL FIX
    env = WoodDetectionRewardWrapper(env)
    env = CameraStabilityWrapper(env)

    if render:
        env = RenderWrapper(env)

    env = PovImageWrapper(env)
    env = ActionWrapper(env)
    env = ResetRetryWrapper(env)


    model = SAC.load(MODEL_PATH, env=env)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        logger.info(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}")
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    evaluate()