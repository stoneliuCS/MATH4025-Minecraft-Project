import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """Encodes a single 64x64 greyscale image frame into a flat feature vector.
 
    Expects input of shape (N, 1, 64, 64) and returns (N, out_dim).
    Designed to be applied across the time dimension via reshaping so that
    the LSTM-based models remain drop-in compatible.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (N, 1, 64, 64) -> (N, 32, 32, 32)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (N, 32, 32, 32) -> (N, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (N, 64, 16, 16) -> (N, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (N, 128, 8, 8) -> (N, 128, 4, 4)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),           # (N, 128 * 4 * 4) = (N, 2048)
            nn.Linear(2048, 256),
            nn.ReLU(),
        )
        self.out_dim = 256
 
    def forward(self, obs_flat):
        """
        Args:
            obs_flat: (T, 64*64) — flattened greyscale frames.
        Returns:
            features: (T, out_dim)
        """
        if len(obs_flat.shape) == 3:
            batch, T, _ = obs_flat.shape
        elif len(obs_flat.shape) == 2:
            T, _ = obs_flat.shape
            batch = 1
        # Reshape to treat every (batch, t) pair as an independent image
        x = obs_flat.view(batch * T, 1, 64, 64)
        x = self.net(x)                         # (batch*T, out_dim)
        if len(obs_flat.shape) == 3:
            return x.view(batch, T, self.out_dim)   # (batch, T, out_dim)
        elif len(obs_flat.shape) == 2:
            return x.view(T, self.out_dim)   # (batch, T, out_dim)
        