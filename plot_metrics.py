"""Plot training metrics from artifacts/training_metrics.json.

Usage:
    python plot_metrics.py                     # default path
    python plot_metrics.py path/to/metrics.json
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


def smooth(values, weight=0.9):
    """Exponential moving average for noisy curves."""
    s = []
    last = values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        s.append(last)
    return s


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/training_metrics.json"
    with open(path) as f:
        m = json.load(f)

    eps = m["episodes"]
    if not eps:
        print("No episodes recorded yet.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MineRL Wood Gathering — Training Metrics", fontsize=14, fontweight="bold")

    # ── Episode Reward ────────────────────────────────────────────────
    ax = axes[0][0]
    ax.plot(eps, m["episode_rewards"], alpha=0.3, color="steelblue", label="Raw")
    ax.plot(eps, smooth(m["episode_rewards"]), color="steelblue", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Logs Collected ────────────────────────────────────────────────
    ax = axes[0][1]
    ax.bar(eps, m["logs_collected"], color="sienna", alpha=0.6, label="Per Episode")
    ax2 = ax.twinx()
    ax2.plot(eps, m["cumulative_logs"], color="darkgreen", linewidth=2, label="Cumulative")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Logs (per ep)")
    ax2.set_ylabel("Cumulative Logs")
    ax.set_title("Logs Collected")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    ax.grid(True, alpha=0.3)

    # ── Episode Length ────────────────────────────────────────────────
    ax = axes[1][0]
    ax.plot(eps, m["episode_lengths"], alpha=0.3, color="coral")
    ax.plot(eps, smooth(m["episode_lengths"]), color="coral", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)

    # ── Attack Ratio ──────────────────────────────────────────────────
    ax = axes[1][1]
    ax.plot(eps, m["attack_ratio"], alpha=0.3, color="mediumpurple")
    ax.plot(eps, smooth(m["attack_ratio"]), color="mediumpurple", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Attack %")
    ax.set_title("Attack Ratio per Episode")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = path.replace(".json", ".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()