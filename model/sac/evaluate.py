import logging
from stable_baselines3 import SAC
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
MODEL_PATH = "artifacts/sac_final.zip"

def evaluate(n_episodes: int = 5):
    env_name = "GatherWood-v0"
    wood_env = GatherWoodEnvironment()
    wood_env.register()

    env = create_environment(env_name, interactive=True)
    env = LogRewardWrapper(env)
    env = StickyAttackWrapper(env, sticky_ticks=5)
    env = WoodDetectionRewardWrapper(env)
    env = RenderWrapper(env)
    env = PovImageWrapper(env)
    env = ActionWrapper(env)

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