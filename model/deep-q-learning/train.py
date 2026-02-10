import os
import sys
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow imports from the same directory (needed because "deep-q-learning"
# contains a hyphen and cannot be a regular Python package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dqn import DQN
from replay_buffer import ReplayBuffer
from wrappers import GrayscaleWrapper, FrameStackWrapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 1_000    # steps
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50_000
WARMUP_STEPS = 10_000
N_FRAMES = 4
N_EPISODES = 200
MAX_STEPS_PER_EPISODE = 2_000
CHECKPOINT_FREQ = 10          # episodes
CHECKPOINT_PATH = "artifacts/dqn_model.pt"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, n_actions, device):
        self.n_actions = n_actions
        self.device = device

        self.policy_net = DQN(N_FRAMES, n_actions).to(device)
        self.target_net = DQN(N_FRAMES, n_actions).to(device)
        self.update_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()

        self.total_steps = 0

    # -- epsilon schedule --------------------------------------------------
    @property
    def epsilon(self):
        frac = min(1.0, self.total_steps / EPSILON_DECAY_STEPS)
        return EPSILON_START + frac * (EPSILON_END - EPSILON_START)

    # -- action selection ---------------------------------------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.from_numpy(state.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
            return self.policy_net(t).argmax(dim=1).item()

    # -- one gradient step --------------------------------------------------
    def optimize(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + GAMMA * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    # -- target update ------------------------------------------------------
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_dqn(env):
    """Main DQN training loop.

    *env* should already be wrapped with RestrictedActionWrapper.
    This function adds GrayscaleWrapper + FrameStackWrapper on top.
    """
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, N_FRAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    agent = DQNAgent(n_actions=4, device=device)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, frame_shape=(N_FRAMES, 64, 64))

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    all_rewards = []

    for episode in range(1, N_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.total_steps += 1

            # Train after warmup
            if agent.total_steps >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = agent.optimize(batch)
                episode_loss += loss
                loss_count += 1

            # Target network update
            if agent.total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            if done:
                break

        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        all_rewards.append(episode_reward)
        logger.info(
            f"Episode {episode:>3d}/{N_EPISODES} | "
            f"Reward: {episode_reward:>8.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Buffer: {len(buffer):>6d} | "
            f"Steps: {agent.total_steps}"
        )

        if episode % CHECKPOINT_FREQ == 0:
            torch.save(agent.policy_net.state_dict(), CHECKPOINT_PATH)
            logger.info(f"  -> Checkpoint saved to {CHECKPOINT_PATH}")

    # Final save
    torch.save(agent.policy_net.state_dict(), CHECKPOINT_PATH)
    logger.info(f"Training complete. Final model saved to {CHECKPOINT_PATH}")
    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_dqn(env, model_path=CHECKPOINT_PATH, episodes=5, render=False):
    """Load a saved DQN model and run a greedy policy."""
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, N_FRAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(N_FRAMES, n_actions=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < MAX_STEPS_PER_EPISODE:
            if render:
                env.render()
            with torch.no_grad():
                t = torch.from_numpy(state.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                action = model(t).argmax(dim=1).item()
            state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        logger.info(f"Eval episode {ep}/{episodes} | Reward: {total_reward:.2f} | Steps: {step}")

    env.close()
