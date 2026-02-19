import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Actor-Critic CNN (same backbone style as your DQN)
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, n_frames=4, n_actions=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64 * 4 * 4, 512)

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value


# ============================================================
# PPO Agent
# ============================================================

class PPOAgent:
    def __init__(self, n_actions, device):
        self.device = device
        self.model = ActorCritic(n_frames=4, n_actions=n_actions).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4)

        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.ppo_epochs = 4

    # --------------------------------------------------------

    def select_action(self, state):
        state = torch.from_numpy(state.astype("float32") / 255.0)
        state = state.unsqueeze(0).to(self.device)

        logits, value = self.model(state)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.detach()

    # --------------------------------------------------------

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    # --------------------------------------------------------

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        returns = torch.tensor(returns).to(self.device)
        advantages = torch.tensor(advantages).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            logits, values = self.model(states)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
