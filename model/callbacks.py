import logging
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class RewardPlotCallback(BaseCallback):
    """Records episode rewards during training and saves a plot at the end.

    Uses the episode info SB3 writes to locals["infos"] at episode boundaries.
    The plot is saved to `output_path` (default: artifacts/reward_plot.png).
    Compatible with both SAC and PPO.
    """

    def __init__(self, output_path: str = "artifacts/reward_plot.png", window: int = 10, title: str = "Training — Episode Rewards"):
        super().__init__()
        self.output_path = output_path
        self.window = window
        self.title = title
        self.episode_rewards: list[float] = []
        self.episode_timesteps: list[int] = []
        self._load_existing_csv()

    def _load_existing_csv(self) -> None:
        csv_path = self.output_path.replace(".png", ".csv")
        if not os.path.exists(csv_path):
            return
        with open(csv_path) as f:
            next(f)  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ts, r = line.split(",")
                self.episode_timesteps.append(int(ts))
                self.episode_rewards.append(float(r))
        logger.info(f"Resumed {len(self.episode_rewards)} episodes from {csv_path}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))
                self.episode_timesteps.append(self.num_timesteps)
                self._save_rewards()
        return True

    def _save_rewards(self) -> None:
        if not self.episode_rewards:
            return
        csv_path = self.output_path.replace(".png", ".csv")
        with open(csv_path, "w") as f:
            f.write("timestep,episode_reward\n")
            for ts, r in zip(self.episode_timesteps, self.episode_rewards):
                f.write(f"{ts},{r}\n")

    def _plot(self) -> None:
        if not self.episode_rewards:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping reward plot.")
            return

        rewards = np.array(self.episode_rewards)
        timesteps = np.array(self.episode_timesteps)

        if len(rewards) >= self.window:
            kernel = np.ones(self.window) / self.window
            smoothed = np.convolve(rewards, kernel, mode="valid")
            smooth_ts = timesteps[self.window - 1:]
        else:
            smoothed, smooth_ts = rewards, timesteps

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(timesteps, rewards, alpha=0.3, color="steelblue", label="episode reward")
        ax.plot(smooth_ts, smoothed, color="steelblue", linewidth=2,
                label=f"{self.window}-ep rolling avg")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Episode reward")
        ax.set_title(self.title)
        ax.legend()
        fig.tight_layout()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fig.savefig(self.output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Reward plot saved to {self.output_path}")

    def _on_training_end(self) -> None:
        self._save_rewards()
        if not self.episode_rewards:
            logger.warning("No episode data collected — skipping reward plot.")
            return
        self._plot()
