import torch
import torch.nn as nn


class DQN(nn.Module):
    """CNN-based DQN for 64x64 grayscale frame stacks (Nature DQN architecture)."""

    def __init__(self, n_frames=4, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),  # -> 15x15x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),        # -> 6x6x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),        # -> 4x4x64
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # x: (batch, n_frames, 64, 64) as float32 in [0, 1]
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
