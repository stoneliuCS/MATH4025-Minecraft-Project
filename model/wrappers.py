import cv2
import gym
import numpy as np
from collections import deque

FRAME_SIZE = 64


class GrayscaleWrapper(gym.ObservationWrapper):
    """Extract 'pov' from obs dict, convert RGB to grayscale, resize to 64x64."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(FRAME_SIZE, FRAME_SIZE), dtype=np.uint8
        )

    def observation(self, obs):
        if isinstance(obs, dict):
            pov = obs["pov"]
        else:
            pov = obs
        # RGB -> grayscale via luminance weights
        gray = np.dot(pov[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
        gray = gray.astype(np.uint8)
        # Resize to 64x64
        gray = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        return gray


class FrameStackWrapper(gym.Wrapper):
    """Stack the last N grayscale frames into a single observation."""

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_frames, 64, 64), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_stacked()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked(), reward, done, info

    def _get_stacked(self):
        return np.stack(self.frames, axis=0)
