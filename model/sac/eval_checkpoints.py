"""Evaluate a series of SAC checkpoints and plot average reward vs timestep.

Loads each checkpoint zip, runs N episodes against GatherWood-v0, records the
total episode reward, then saves a CSV and a plot.

Usage:
    python -m model.sac.eval_checkpoints [--checkpoint-dir DIR] [--episodes N] [--out DIR]
    make eval-checkpoints

Checkpoints are discovered automatically from --checkpoint-dir, sorted by step count.
"""

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np

from environment.wood_environment import (
    GatherWoodEnvironment,
    LogRewardWrapper,
    StickyAttackWrapper,
    WoodDetectionRewardWrapper,
    PovImageWrapper,
    ActionWrapper,
)
from model.environment import create_environment
from stable_baselines3 import SAC

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _build_env():
    env_name = "GatherWood-v0"
    GatherWoodEnvironment().register()
    env = create_environment(env_name, interactive=True)
    env = LogRewardWrapper(env)
    env = StickyAttackWrapper(env, sticky_ticks=15)
    env = WoodDetectionRewardWrapper(env)
    env = PovImageWrapper(env)
    env = ActionWrapper(env)
    return env


def _discover_checkpoints(checkpoint_dir: str) -> list[tuple[int, Path]]:
    """Return (step_count, path) pairs sorted by step count."""
    pattern = re.compile(r"_(\d+)_steps\.zip$")
    results = []
    for p in Path(checkpoint_dir).glob("*.zip"):
        m = pattern.search(p.name)
        if m:
            results.append((int(m.group(1)), p))
    return sorted(results, key=lambda x: x[0])


def _evaluate_checkpoint(checkpoint_path: Path, env, n_episodes: int) -> list[float]:
    # Load without env so SB3 doesn't wrap it in Monitor/DummyVecEnv and
    # trigger an internal reset before we're ready.
    model = SAC.load(str(checkpoint_path))
    episode_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward
        episode_rewards.append(total)
        logger.info(f"  episode reward: {total:.2f}")
    return episode_rewards


def _save_csv(results: list[tuple[int, float, float]], out_dir: str):
    path = Path(out_dir) / "checkpoint_rewards.csv"
    with open(path, "w") as f:
        f.write("timestep,mean_reward,std_reward\n")
        for step, mean, std in results:
            f.write(f"{step},{mean:.4f},{std:.4f}\n")
    logger.info(f"Saved CSV to {path}")
    return path


def _plot(results: list[tuple[int, float, float]], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot.")
        return

    steps  = [r[0] for r in results]
    means  = [r[1] for r in results]
    stds   = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, means, marker="o", color="steelblue", label="mean reward")
    ax.fill_between(steps,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="steelblue", label="±1 std")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Episode reward")
    ax.set_title("SAC Checkpoint Evaluation")
    ax.legend()
    fig.tight_layout()

    path = Path(out_dir) / "checkpoint_rewards.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot to {path}")


def evaluate(
    checkpoint_dir: str = "artifacts/sac",
    checkpoint: str | None = None,
    episodes: int = 3,
    out_dir: str = "artifacts",
):
    if checkpoint:
        p = Path(checkpoint)
        m = re.compile(r"_(\d+)_steps\.zip$").search(p.name)
        step = int(m.group(1)) if m else 0
        checkpoints = [(step, p)]
    else:
        checkpoints = _discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint zips found in {checkpoint_dir}")

    logger.info(f"Found {len(checkpoints)} checkpoints — evaluating {episodes} episodes each")
    os.makedirs(out_dir, exist_ok=True)

    env = _build_env()
    results = []
    try:
        for step, path in checkpoints:
            logger.info(f"Evaluating {path.name} ({step:,} steps) ...")
            rewards = _evaluate_checkpoint(path, env, episodes)
            mean, std = float(np.mean(rewards)), float(np.std(rewards))
            logger.info(f"  mean={mean:.2f}  std={std:.2f}")
            results.append((step, mean, std))
    finally:
        env.close()

    _save_csv(results, out_dir)
    _plot(results, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="artifacts/sac",
                        help="Directory containing sac_wood_*_steps.zip files")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a single checkpoint zip to evaluate")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes to run per checkpoint")
    parser.add_argument("--out", default="artifacts",
                        help="Directory to save CSV and plot")
    args = parser.parse_args()
    evaluate(checkpoint_dir=args.checkpoint_dir, checkpoint=args.checkpoint, episodes=args.episodes, out_dir=args.out)
