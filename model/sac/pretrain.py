"""Behavioural Cloning pretraining for the SAC policy using MineRL demonstrations.

Loads human gameplay data from a downloaded MineRL dataset directory and trains
the SAC actor to imitate demonstrated actions.  No Minecraft process is
required — a lightweight dummy env supplies the correct spaces.

Expected dataset layout:
    data/MineRLTreechop-v0/MineRLTreechop-v0/<trajectory>/
        recording.mp4   — 64×64 gameplay video
        rendered.npz    — action arrays (action$camera, action$forward, …)

Usage:
    python -m model.main --mode sac-pretrain [--data-dir data/] [--epochs 5]
    make sac-pretrain
"""

import logging
import os
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import gym
import gym.spaces
import numpy as np
import torch
import torch.optim as optim
from stable_baselines3 import SAC

from environment.wood_environment import (
    FRAME_SIZE,
    CAMERA_MAX_ANGLE,
    ACTION_DIM,
)

logger = logging.getLogger(__name__)

BC_EPOCHS = 5
BC_BATCH_SIZE = 256
BC_LR = 1e-4
PRETRAIN_MODEL_PATH = "artifacts/sac_pretrained.zip"
# Dataset lives at data/{ENV}/{ENV}/ (double-directory layout from the download)
MINERL_ENV_NAME = "MineRLTreechop-v0"


# ── dummy environment ─────────────────────────────────────────────────────────

class _DummyWoodEnv(gym.Env):  # pyright: ignore[reportPrivateImportUsage]
    """Spaces match GatherWood-v0 after PovImageWrapper + ActionWrapper."""

    metadata: dict = {"render.modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32,
        )

    def reset(self):
        return self.observation_space.sample()

    def step(self, _action):
        return self.observation_space.sample(), 0.0, False, {}

    def render(self, _mode="human"):
        pass

    def close(self):
        pass


# ── dataset loading ───────────────────────────────────────────────────────────

def _iter_trajectories(data_dir: str, env_name: str) -> Iterator[Path]:
    """Yield all trajectory directories.

    Handles both flat layout (data_dir/env_name/<traj>/) and the double-directory
    layout produced by the downloaded dataset (data_dir/env_name/env_name/<traj>/).
    """
    env_path = Path(data_dir) / env_name
    if not env_path.exists():
        raise FileNotFoundError(f"Dataset not found at {env_path}")

    # Double-directory layout: data/MineRLTreechop-v0/MineRLTreechop-v0/<traj>/
    nested = env_path / env_name
    search_root = nested if nested.exists() else env_path

    for traj in sorted(search_root.iterdir()):
        if traj.is_dir():
            yield traj


def _load_trajectory(
    traj_dir: Path,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Load one trajectory from rendered.npz + recording.mp4.

    Returns:
        obs:   (T, 3, FRAME_SIZE, FRAME_SIZE) uint8
        acts:  (T, ACTION_DIM)                float32
    or None if the trajectory is incomplete / unreadable.
    """
    video_path = traj_dir / "recording.mp4"
    npz_path   = traj_dir / "rendered.npz"

    if not video_path.exists() or not npz_path.exists():
        logger.debug(f"Skipping incomplete trajectory: {traj_dir.name}")
        return None

    # ── actions ──────────────────────────────────────────────────────────────
    data    = np.load(npz_path, allow_pickle=True)
    camera  = data["action$camera"].astype(np.float32)   # (T, 2)
    forward = data["action$forward"].astype(np.float32)  # (T,)
    back    = data["action$back"].astype(np.float32)
    left    = data["action$left"].astype(np.float32)
    right   = data["action$right"].astype(np.float32)
    attack  = data["action$attack"].astype(np.float32)

    T = len(forward)
    if T == 0:
        return None

    pitch = np.clip(camera[:, 0] / CAMERA_MAX_ANGLE, -1.0, 1.0)
    yaw   = np.clip(camera[:, 1] / CAMERA_MAX_ANGLE, -1.0, 1.0)

    def _binary(arr):
        return np.where(arr > 0, 0.5, -0.5).astype(np.float32)

    acts = np.stack(
        [pitch, yaw, _binary(forward), _binary(back),
         _binary(left), _binary(right), _binary(attack)],
        axis=1,
    )  # (T, 7)

    # ── frames ───────────────────────────────────────────────────────────────
    # Video is already 64×64; just convert BGR→RGB and transpose to (C,H,W)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(np.transpose(frame, (2, 0, 1)))  # (C, H, W)
    cap.release()

    if not frames:
        return None

    obs = np.stack(frames, axis=0).astype(np.uint8)  # (T_vid, 3, H, W)

    # Align lengths (video decoder may produce slightly more/fewer frames)
    n = min(len(obs), T)
    return obs[:n], acts[:n]


def _batch_iter(
    data_dir: str,
    env_name: str,
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (obs, acts) batches across all trajectories."""
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []

    for traj_dir in _iter_trajectories(data_dir, env_name):
        result = _load_trajectory(traj_dir)
        if result is None:
            continue
        obs, acts = result
        obs_buf.append(obs)
        act_buf.append(acts)

        combined_obs  = np.concatenate(obs_buf,  axis=0)
        combined_acts = np.concatenate(act_buf, axis=0)

        while len(combined_obs) >= batch_size:
            yield combined_obs[:batch_size], combined_acts[:batch_size]
            combined_obs  = combined_obs[batch_size:]
            combined_acts = combined_acts[batch_size:]

        obs_buf  = [combined_obs]  if len(combined_obs)  else []
        act_buf  = [combined_acts] if len(combined_acts) else []

    # Yield remaining partial batch
    if obs_buf:
        combined_obs  = np.concatenate(obs_buf,  axis=0)
        combined_acts = np.concatenate(act_buf, axis=0)
        if len(combined_obs) > 0:
            yield combined_obs, combined_acts


# ── BC training ───────────────────────────────────────────────────────────────

def _bc_loss(model: SAC, obs_np: np.ndarray, acts_np: np.ndarray) -> torch.Tensor:
    """Negative log-likelihood of demo actions under the current actor."""
    device = model.policy.device
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device) / 255.0
    act_t = torch.tensor(acts_np, dtype=torch.float32, device=device)

    mean, log_std, _ = model.policy.actor.get_action_dist_params(obs_t)
    dist = model.policy.actor.action_dist.proba_distribution(mean, log_std)
    return -dist.log_prob(act_t).mean()


def pretrain(
    data_dir: str = "data",
    env_name: str = MINERL_ENV_NAME,
    epochs: int = BC_EPOCHS,
    batch_size: int = BC_BATCH_SIZE,
    lr: float = BC_LR,
    output_path: str = PRETRAIN_MODEL_PATH,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_env = _DummyWoodEnv()
    model = SAC("CnnPolicy", dummy_env, verbose=0, buffer_size=1000)
    optimizer = optim.Adam(model.policy.actor.parameters(), lr=lr)
    model.policy.actor.train()

    logger.info(f"BC pretraining — {epochs} epoch(s), batch {batch_size}, lr {lr}")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_steps = 0

        for obs_b, acts_b in _batch_iter(data_dir, env_name, batch_size):
            optimizer.zero_grad()
            loss = _bc_loss(model, obs_b, acts_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1
            if n_steps % 100 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}  step {n_steps}  "
                    f"avg BC loss: {epoch_loss / n_steps:.4f}"
                )

        logger.info(f"Epoch {epoch}/{epochs} — avg BC loss: {epoch_loss / max(n_steps, 1):.4f}")

    model.save(output_path)
    logger.info(f"Pretrained model saved to {output_path}")
    dummy_env.close()
    return output_path
