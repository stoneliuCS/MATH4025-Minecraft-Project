"""
Plot monitor CSV files from training runs.
Produces:
  - Individual plots for each run folder
  - Side-by-side comparison plot of all runs
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")

FOLDERS = [
    "sac_ray_tracing_main_v2",
    "ppo_ray_tracing_main_v1",
    "sac_ray_tracing_rewards_fine_tuned_v6",
    "ppo_ray_tracing_taiga_v1",
    "sac_ray_tracing_taiga_v1",
]

LABELS = {
    "sac_ray_tracing_main_v2": "SAC Main v2",
    "ppo_ray_tracing_main_v1": "PPO Main v1",
    "sac_ray_tracing_rewards_fine_tuned_v6": "SAC Fine-tuned v6",
    "ppo_ray_tracing_taiga_v1": "PPO Taiga v1",
    "sac_ray_tracing_taiga_v1": "SAC Taiga v1",
}

WINDOW = 10  # rolling average window (episodes)


def load_monitor_files(folder_path):
    """Load and concatenate all monitor CSVs from a folder, sorted by t_start."""
    files = sorted(glob.glob(os.path.join(folder_path, "*.monitor.csv")))
    dfs = []
    for f in files:
        with open(f) as fh:
            first_line = fh.readline().strip()
        # Parse t_start from header comment
        t_start = 0.0
        if "t_start" in first_line:
            import json
            meta = json.loads(first_line.lstrip("#").strip())
            t_start = meta.get("t_start", 0.0)
        df = pd.read_csv(f, comment="#")
        df.columns = [c.strip() for c in df.columns]
        df["t_start"] = t_start
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["r", "l", "t"])
    combined = pd.concat(dfs, ignore_index=True)
    # Sort by absolute timestamp
    combined["t_abs"] = combined["t_start"] + combined["t"]
    combined = combined.sort_values("t_abs").reset_index(drop=True)
    combined["episode"] = range(1, len(combined) + 1)
    combined["timestep"] = combined["l"].cumsum()
    return combined


def smooth(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def plot_individual(folder, df, output_dir):
    label = LABELS[folder]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{label} — Episode Reward", fontsize=13, fontweight="bold")

    ts = df["timestep"]
    ax.plot(ts, df["r"], alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
    ax.plot(ts, smooth(df["r"], WINDOW), color="steelblue", linewidth=1.8,
            label=f"rolling mean ({WINDOW} ep)")
    ax.set_ylabel("Episode Reward")
    ax.set_xlabel("Timestep")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}K"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{folder}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison(all_data, output_dir, truncate=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    min_steps = min(df["timestep"].iloc[-1] for df in all_data.values() if not df.empty)

    title_suffix = f"(truncated to {min_steps/1e6:.2f}M steps)" if truncate else "(full length)"
    fname = "comparison_truncated.png" if truncate else "comparison_full.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"All Runs — Episode Reward Comparison {title_suffix}",
                 fontsize=13, fontweight="bold")

    for i, folder in enumerate(FOLDERS):
        df = all_data[folder]
        if df.empty:
            continue
        if truncate:
            df = df[df["timestep"] <= min_steps]
        label = LABELS[folder]
        color = colors[i % len(colors)]
        ts = df["timestep"]
        ax.plot(ts, df["r"], alpha=0.15, color=color, linewidth=0.7)
        ax.plot(ts, smooth(df["r"], WINDOW), color=color, linewidth=1.8, label=label)

    ax.set_xlabel("Timestep")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}K"))
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


LOG_REWARD = 2.0
TIME_PENALTY_PER_EP = 0.0002 * 2000  # 0.4 per episode


def plot_avg_wood(all_data, output_dir):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels, values, bar_colors = [], [], []
    for i, folder in enumerate(FOLDERS):
        df = all_data[folder]
        if df.empty:
            continue
        avg_wood = max(0.0, df["r"].mean() + TIME_PENALTY_PER_EP) / LOG_REWARD
        labels.append(LABELS[folder])
        values.append(avg_wood)
        bar_colors.append(colors[i % len(colors)])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Average Wood Mined per Episode", fontsize=13, fontweight="bold")

    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Avg Wood / Episode  (reward ÷ 2.0)")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "avg_wood_per_episode.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


SAC_FOLDERS = [f for f in FOLDERS if f.startswith("sac")]
PPO_FOLDERS = [f for f in FOLDERS if f.startswith("ppo")]


def plot_variance(all_data, output_dir):
    """Plot reward mean ± std band for SAC (combined) vs PPO (combined) over episodes."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    fig.suptitle("Reward Variance: SAC vs PPO (Combined Runs)", fontsize=13, fontweight="bold")

    groups = [
        ("SAC", SAC_FOLDERS, "#89b4fa"),
        ("PPO", PPO_FOLDERS, "#a6e3a1"),
    ]

    for ax, (name, folders, color) in zip(axes, groups):
        # Align all runs to episode index, interpolate onto common episode grid
        series_list = []
        for folder in folders:
            df = all_data[folder]
            if df.empty:
                continue
            series_list.append(df["r"].values)

        # Pad shorter runs with NaN so we can stack
        max_len = max(len(s) for s in series_list)
        padded = np.full((len(series_list), max_len), np.nan)
        for i, s in enumerate(series_list):
            padded[i, :len(s)] = s

        episodes = np.arange(1, max_len + 1)
        mean = np.nanmean(padded, axis=0)
        std  = np.nanstd(padded, axis=0)
        mean_smooth = pd.Series(mean).rolling(WINDOW, min_periods=1).mean().values
        std_smooth  = pd.Series(std).rolling(WINDOW, min_periods=1).mean().values

        # Per-run raw traces
        for i, folder in enumerate(folders):
            df = all_data[folder]
            if df.empty:
                continue
            ax.plot(df["episode"], df["r"], alpha=0.1, color=color, linewidth=0.6)
            ax.plot(df["episode"], smooth(df["r"], WINDOW),
                    alpha=0.4, color=color, linewidth=1.0, linestyle="--",
                    label=LABELS[folder])

        # Combined mean ± std band
        ax.fill_between(episodes,
                        mean_smooth - std_smooth,
                        mean_smooth + std_smooth,
                        alpha=0.25, color=color, label="±1 std")
        ax.plot(episodes, mean_smooth, color=color, linewidth=2.2, label=f"{name} mean")

        ax.set_title(f"{name} — {len(folders)} run{'s' if len(folders) > 1 else ''}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "variance_sac_vs_ppo.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = {}

    print("Loading monitor files...")
    for folder in FOLDERS:
        folder_path = os.path.join(DATA_DIR, folder)
        df = load_monitor_files(folder_path)
        all_data[folder] = df
        print(f"  {LABELS[folder]}: {len(df)} episodes")

    print("\nGenerating individual plots...")
    for folder, df in all_data.items():
        if df.empty:
            print(f"  Skipping {folder} (no data)")
            continue
        plot_individual(folder, df, OUTPUT_DIR)

    print("\nGenerating comparison plots...")
    plot_comparison(all_data, OUTPUT_DIR, truncate=False)
    plot_comparison(all_data, OUTPUT_DIR, truncate=True)

    print("\nGenerating avg wood plot...")
    plot_avg_wood(all_data, OUTPUT_DIR)

    print("\nGenerating SAC vs PPO variance plot...")
    plot_variance(all_data, OUTPUT_DIR)

    print("\nDone. Plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
