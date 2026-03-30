import torch 
import torch.nn as nn

class LSTMRewardModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)  # scalar reward output

    def forward(self, segment):
        # segment: (batch, timesteps, obs_dim)
        lstm_out, _ = self.lstm(segment)
        # Sum reward over timesteps (as in Christiano et al.)
        r = self.fc(lstm_out).squeeze(-1)   # (batch, timesteps)
        return r.sum(dim=1)                 # (batch,) — cumulative reward per segment
