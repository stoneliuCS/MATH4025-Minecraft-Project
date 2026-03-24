"""SB3 callback that logs training metrics to a JSON file for plotting."""
import json
import os
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

LOG_ITEMS = ["oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log"]


class TrainingMetricsCallback(BaseCallback):
    """Logs per-episode and periodic metrics to artifacts/training_metrics.json.

    Tracked metrics:
      - episode_reward: total reward per episode
      - episode_length: steps per episode
      - logs_collected: inventory log count at episode end
      - cumulative_logs: running total across all episodes
      - wall_time: seconds since training start
      - mining_ticks: steps where the agent was attacking nearby wood
      - attack_ratio: fraction of steps where the agent attacked
    """

    def __init__(self, save_dir: str = "artifacts", save_freq: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq  # save every N episodes

        self.metrics = {
            "episodes": [],
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "logs_collected": [],
            "cumulative_logs": [],
            "wall_time": [],
            "mining_ticks": [],
            "attack_ratio": [],
        }

        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_attacks = 0
        self._ep_mining = 0
        self._ep_count = 0
        self._cumulative_logs = 0
        self._start_time = None

    def _on_training_start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._start_time = time.time()

    def _on_step(self) -> bool:
        self._ep_length += 1

        # Track attacks from the action
        actions = self.locals.get("actions")
        if actions is not None:
            action = actions[0]
            if hasattr(action, "__len__") and len(action) >= 4:
                if action[3] > 0:
                    self._ep_attacks += 1

        # Track reward
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._ep_reward += float(rewards[0])

        # Check for episode end
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            self._ep_count += 1

            # Try to get log count from the last observation
            ep_logs = 0
            infos = self.locals.get("infos")
            if infos and isinstance(infos[0], dict):
                obs = infos[0].get("terminal_observation")
                if isinstance(obs, dict) and "inventory" in obs:
                    inv = obs["inventory"]
                    if isinstance(inv, dict):
                        ep_logs = sum(int(inv.get(item, 0)) for item in LOG_ITEMS)

            self._cumulative_logs += ep_logs

            self.metrics["episodes"].append(self._ep_count)
            self.metrics["timesteps"].append(self.num_timesteps)
            self.metrics["episode_rewards"].append(round(self._ep_reward, 4))
            self.metrics["episode_lengths"].append(self._ep_length)
            self.metrics["logs_collected"].append(ep_logs)
            self.metrics["cumulative_logs"].append(self._cumulative_logs)
            self.metrics["wall_time"].append(round(time.time() - self._start_time, 1))
            self.metrics["mining_ticks"].append(self._ep_mining)
            atk_ratio = self._ep_attacks / max(self._ep_length, 1)
            self.metrics["attack_ratio"].append(round(atk_ratio, 4))

            if self.verbose >= 1:
                print(f"[EP {self._ep_count}] reward={self._ep_reward:.2f} "
                      f"len={self._ep_length} logs={ep_logs} "
                      f"cumul={self._cumulative_logs} atk%={atk_ratio:.1%}")

            # Save periodically
            if self._ep_count % self.save_freq == 0:
                self._save()

            # Reset episode trackers
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_attacks = 0
            self._ep_mining = 0

        return True

    def _on_training_end(self):
        self._save()

    def _save(self):
        path = os.path.join(self.save_dir, "training_metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics, f)