import torch.nn as nn 

# --- Small value network (critic) for PPO advantage estimation
class LSTMValue(nn.Module):
    """Critic: maps observations to a scalar state-value estimate."""
    def __init__(self, obs_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, 1)
 
    def forward(self, obs, hidden=None):
        # obs: (batch, 1, obs_dim) — we feed one step at a time during rollout
        out, hidden = self.lstm(obs, hidden)
        value = self.fc(out).squeeze(-1)   # (batch, 1)
        return value, hidden