import os
import sys
import random
import logging

import mlflow

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

from reward_wrappers.east_wrapper import EastActionWrapper

logger = logging.getLogger(__name__)

mlflow.set_experiment("Q Learning East")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_ACTIONS = 10
LEARNING_RATE = 0.00001
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 100    # steps
GAMMA = 0.8
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 5000
WARMUP_STEPS = 10
N_FRAMES = 6
N_EPISODES = 200
MAX_STEPS_PER_EPISODE = 750
CHECKPOINT_FREQ = 5          # episodes
CHECKPOINT_PATH = "artifacts/dqn_model.pt"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, device):
        self.n_actions = N_ACTIONS
        self.device = device

        self.policy_net = DQN(N_FRAMES, N_ACTIONS).to(device)
        self.target_net = DQN(N_FRAMES, N_ACTIONS).to(device)
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

    def select_action(self, state):
        # select an action 
        # the Q network is being inferenced in this state
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            # normalize the state, reshape it so that it will have a batch dimension, and then move it on to the GPU
            state = torch.from_numpy(state.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
            logger.debug(f"shape of state tensor: {state.shape}")
            # inference the policy network on the state 
            return self.policy_net(state).argmax(dim=1).item()

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

    # add the wrapper to the environment which will generate the rewards
    env = EastActionWrapper(env)

    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, N_FRAMES)

    # connect to GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # create agent
    agent = DQNAgent(device=device)
    # load the saved checkpoint if it exists
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Loading model checkpoint from: {CHECKPOINT_PATH}")
        agent.target_net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        agent.policy_net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    # create buffer to store recent states 
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, frame_shape=(N_FRAMES, 64, 64))

    # make sure that the directory where we want to store the checkpoint file exists
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    all_rewards = []

    # this is for logging purposes, signify that a new run is starting
    with mlflow.start_run():

        # log the parameters for this run: 
        mlflow.log_params(
            {
                "learning_rate" : LEARNING_RATE, 
                "gamma" : GAMMA, 
                "target_update_freq" : TARGET_UPDATE_FREQ, 
                "epsilon_start" : EPSILON_START,
                "epsilon_end" : EPSILON_END,
                "epsilon_decay_steps" : EPSILON_DECAY_STEPS, 
                "warmup_steps" : WARMUP_STEPS,
                "n_frames" : N_FRAMES, 
                "max_steps_per_episode" : MAX_STEPS_PER_EPISODE, 
                "checkpoint_freq" : CHECKPOINT_FREQ,
                "checkpoint_path" : CHECKPOINT_PATH
            }
        )


        for episode in range(1, N_EPISODES + 1):
            state = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                env.render()

                
                # inference the Q neural network to select an action
                action = agent.select_action(state)
                # take the action in the environment, get the next state and the reward
                next_state, reward, done, info = env.step(action)
                # add the information for this step to the state buffer
                buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                agent.total_steps += 1

                # Train after warmup
                if agent.total_steps >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
                    logger.info(f"doing optimization process...")
                    batch = buffer.sample(BATCH_SIZE)
                    loss = agent.optimize(batch)
                    episode_loss += loss
                    loss_count += 1

                # Target network update
                if agent.total_steps % TARGET_UPDATE_FREQ == 0:
                    agent.update_target()

                if done:
                    break
                
                x = info['xpos']
                z = info['zpos']
                logger.info(f"agent.total_steps: {agent.total_steps}, action: {action}, x: {x}, z: {z}, reward: {reward}")

            # give the information for this episode
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

            # log the model's performance on mlflow
            mlflow.log_metrics(
                {"Average Loss": avg_loss, "Reward": reward}, step=episode
            )

            if episode % CHECKPOINT_FREQ == 0:
                torch.save(agent.policy_net.state_dict(), CHECKPOINT_PATH)
                logger.info(f"  -> Checkpoint saved to {CHECKPOINT_PATH}")

    # save a checkpoint for the current model weights
    torch.save(agent.policy_net.state_dict(), CHECKPOINT_PATH)
    logger.info(f"Training complete. Final model saved to {CHECKPOINT_PATH}")
    env.close()
    return agent


