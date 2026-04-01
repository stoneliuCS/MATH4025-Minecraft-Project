import torch 
import torch.nn as nn
from cnn_encoder import CNNEncoder

class LSTMRewardModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64, num_layers=4):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(
            input_size=self.encoder.out_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, 1)  # scalar reward output

    def forward(self, segment):
        """
        Args:
            segment: (batch, T, 64*64) — flattened greyscale frames.
        Returns:
            rewards: (batch,) — cumulative reward per segment (as in Christiano et al.).
        """
        # segment: (batch, timesteps, obs_dim)
        x = self.encoder(segment)
        lstm_out, _ = self.lstm(x)
        # Sum reward over timesteps (as in Christiano et al.)
        r = self.fc(lstm_out).squeeze(-1)   # (batch, timesteps)
        return r.sum(dim=1)                 # (batch,) — cumulative reward per segment
