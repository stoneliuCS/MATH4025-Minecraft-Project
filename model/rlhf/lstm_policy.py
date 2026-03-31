import torch 
import torch.nn as nn 
import numpy as np
from cnn_encoder import CNNEncoder

# LSTM policy model 
# this is meant to take in a set of observations, and generate predictions. 
class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, num_layers=4):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden=None):
        """
        Args:
            obs:    (batch, T, 64*64) — flattened greyscale frames.
            hidden: Optional LSTM hidden state tuple.
        Returns:
            logits: (batch, T, action_dim)
            hidden: Updated LSTM hidden state tuple.
        """
        x = self.encoder(obs)
        out, hidden = self.lstm(obs, hidden)
        return self.fc(out), hidden  # (batch, T, action_dim)
