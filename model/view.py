"""Run a trained SAC or PPO model interactively for viewing — no gradient updates.

Usage:
    python -m model.view --algo ppo --checkpoint artifacts/ppo/ppo_wood_500000_steps.zip
    python -m model.view --algo sac --checkpoint artifacts/sac/sac_wood_532000_steps.zip
"""

import argparse
import logging
import os

from environment.wood_env2 import GatherWoodEnvironment
from model.environment import make_wood_env
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_ppo_env(env_name: str, checkpoint_dir: str):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

    env = DummyVecEnv([lambda: GymV21CompatibilityV0(
        env_id=env_name,
        env=make_wood_env(env_name, render=True, interactive=True),
    )])
    env = VecFrameStack(env, n_stack=4)

    vecnorm_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        logger.info(f"Loaded VecNormalize from {vecnorm_path}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)

    return env


def _build_sac_env(env_name: str):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    env = DummyVecEnv([lambda: GymV21CompatibilityV0(
        env_id=env_name,
        env=make_wood_env(env_name, render=True, interactive=True),
    )])
    env = VecFrameStack(env, n_stack=4)
    return env


def view(algo: str, checkpoint: str, episodes: int = 0):
    """
    Run the model indefinitely (episodes=0) or for a fixed number of episodes.
    Actions are deterministic — no exploration, no gradient updates.
    """
    env_name = "GatherWood-v0"
    GatherWoodEnvironment().register()

    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint))

    if algo == "ppo":
        from stable_baselines3 import PPO
        env = _build_ppo_env(env_name, checkpoint_dir)
        model = PPO.load(checkpoint, env=env, device="auto")
    elif algo == "sac":
        from stable_baselines3 import SAC
        env = _build_sac_env(env_name)
        model = SAC.load(checkpoint, env=env, device="auto")
    else:
        raise ValueError(f"Unknown algo: {algo}")

    logger.info(f"Loaded {algo.upper()} checkpoint: {checkpoint}")
    logger.info("Running in deterministic mode — no training.")

    ep = 0
    try:
        while episodes == 0 or ep < episodes:
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total += float(reward)
            ep += 1
            logger.info(f"Episode {ep} reward: {total:.2f}")
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "ppo"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .zip")
    parser.add_argument("--episodes", type=int, default=0, help="Episodes to run (0 = infinite)")
    args = parser.parse_args()
    view(algo=args.algo, checkpoint=args.checkpoint, episodes=args.episodes)
