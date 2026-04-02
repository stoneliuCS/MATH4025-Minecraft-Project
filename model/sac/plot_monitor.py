"""Plot episode rewards from SB3 monitor CSV files.

Usage:
    python -m model.sac.plot_monitor --monitor-dir data/results/sac_ray_tracing_rewards_fine_tuned_v6
    python -m model.sac.plot_monitor --monitor-dir data/results/sac_ray_tracing_rewards_fine_tuned_v6 --out data/results/reward.png
"""

import argparse
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_monitor_files(monitor_dir: str):
    """Load and concatenate all monitor CSV files in directory, sorted by t_start."""
    pattern = os.path.join(monitor_dir, "*.monitor.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .monitor.csv files found in {monitor_dir}")

    all_rewards = []
    all_lengths = []
    all_t = []

    for path in files:
        with open(path) as f:
            # First line is JSON metadata: #{"t_start": ..., "env_id": ...}
            meta_line = f.readline().strip().lstrip("#")
            import json
            meta = json.loads(meta_line)
            t_start = meta.get("t_start", 0.0)

            # Second line is header: r,l,t
            f.readline()

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                r, l, t = float(parts[0]), int(parts[1]), float(parts[2])
                all_rewards.append(r)
                all_lengths.append(l)
                all_t.append(t_start + t)

    # Sort by wall-clock time
    order = np.argsort(all_t)
    rewards = np.array(all_rewards)[order]
    lengths = np.array(all_lengths)[order]

    # Reconstruct cumulative timesteps from episode lengths
    timesteps = np.cumsum(lengths)

    return timesteps, rewards


def plot(timesteps, rewards, out_path: str, window: int = 10):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(timesteps, rewards, alpha=0.25, color="steelblue", linewidth=0.8, label="episode reward")

    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_ts = timesteps[window - 1:]
        ax.plot(smooth_ts, smoothed, color="steelblue", linewidth=2,
                label=f"{window}-ep rolling avg")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode reward")
    ax.set_title("SAC Training — Episode Rewards (from monitor files)")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")
    print(f"Episodes: {len(rewards)}")
    print(f"Best episode: {rewards.max():.2f} at timestep {timesteps[rewards.argmax()]}")
    print(f"Last 20 avg: {rewards[-20:].mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-dir", required=True, help="Directory containing .monitor.csv files")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <monitor-dir>/reward_monitor_plot.png)")
    parser.add_argument("--window", type=int, default=10, help="Rolling average window size")
    args = parser.parse_args()

    out = args.out or os.path.join(args.monitor_dir, "reward_monitor_plot.png")
    timesteps, rewards = load_monitor_files(args.monitor_dir)
    plot(timesteps, rewards, out, window=args.window)
