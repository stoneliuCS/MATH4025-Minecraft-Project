import torch 
import torch.nn as nn 
import numpy as np

# LSTM policy model 
# this is meant to take in a set of observations, and generate predictions. 
class LSTMPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden=None):
        # obs: (batch, T, obs_dim)
        out, hidden = self.lstm(obs, hidden)
        return self.fc(out), hidden  # (batch, T, action_dim)