"""Plot per-episode reward curves for Q-learning and DQN."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_rewards(path):
    with open(path) as f:
        return json.load(f)


def rolling_average(rewards, window=5):
    if len(rewards) < window:
        return rewards
    return np.convolve(rewards, np.ones(window) / window, mode="valid").tolist()


def plot_single(rewards, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = list(range(1, len(rewards) + 1))
    ax.plot(episodes, rewards, alpha=0.4, label="Per episode")

    smoothed = rolling_average(rewards)
    offset = len(rewards) - len(smoothed)
    ax.plot(range(offset + 1, len(rewards) + 1), smoothed, linewidth=2, label=f"Rolling avg (window=5)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    os.makedirs("artifacts", exist_ok=True)

    q_path = "artifacts/q_learning_rewards.json"
    dqn_path = "artifacts/dqn_rewards.json"

    if os.path.exists(q_path):
        rewards = load_rewards(q_path)
        plot_single(rewards, "Q-Learning: Episode Rewards", "artifacts/q_learning_rewards.png")
    else:
        print(f"Skipping Q-learning (no {q_path})")

    if os.path.exists(dqn_path):
        rewards = load_rewards(dqn_path)
        plot_single(rewards, "DQN: Episode Rewards", "artifacts/dqn_rewards.png")
    else:
        print(f"Skipping DQN (no {dqn_path})")

    # Combined plot if both exist
    if os.path.exists(q_path) and os.path.exists(dqn_path):
        q_rewards = load_rewards(q_path)
        dqn_rewards = load_rewards(dqn_path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(q_rewards) + 1), rolling_average(q_rewards, 3), label="Q-Learning")
        ax.plot(range(1, len(dqn_rewards) + 1), rolling_average(dqn_rewards, 3), label="DQN")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward (smoothed)")
        ax.set_title("Q-Learning vs DQN: Episode Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("artifacts/reward_comparison.png", dpi=150)
        plt.close(fig)
        print("Saved: artifacts/reward_comparison.png")


if __name__ == "__main__":
    main()
