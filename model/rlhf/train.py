import os 
import sys
import random
import logging

import mlflow

import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim


from model import Model
from wrappers import GrayscaleWrapper, FrameStackWrapper
from lstm_policy import LSTMPolicy
from lstm_reward import LSTMRewardModel
from preference_loss import preference_loss
from rlhf_wrapper import RLHFActionWrapper


N_ACTIONS = 10
LEARNING_RATE = 0.00001
BATCH_SIZE = 32
N_FRAMES = 6
N_EPISODES = 200
MAX_STEPS_PER_EPISODE = 750
CHECKPOINT_FREQ = 5          # episodes
CHECKPOINT_PATH = "artifacts/dqn_model.pt"
N_REWARD_EPOCHS = 50
POLICY_LR = 1e-3
REWARD_MODEL_LR = 1e-3


def preprocess_state(state):
    state = torch.from_numpy(state.astype(np.float32) / 255.0).unsqueeze(0)
    return state

def collect_segment(policy, env):
    # act out a segment using the policy, and allow for the human viewer to view for judgement

    # hidden state for the LSTM
    hidden = None
    # history of observations which is used by the reward model
    obs_list = []

    # restart the environment
    state = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        # render the frame for the
        env.render()

        obs = preprocess_state(state)
        with torch.no_grad():
            logits, hidden = policy(obs, hidden)
        action = torch.distributions.Categorical(logits = logits.squeeze()).sample().item()
        obs_list.append(obs)
        next_state, reward, done, info = env.step(action)
        state = next_state
    
    segment = torch.tensor(np.array(obs_list), dtype = torch.float32).unsqueeze(0)
    return segment

def get_human_preference():
    preferred_segment = input("Which segment did you prefer (0 or 1)?")
    return torch.tensor([float(preferred_segment)])


def train(env):
    '''
    
        The main training loop for DPO model
    
    '''

    env = RLHFActionWrapper(env)
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, N_FRAMES)

    # --- Initialize the policy and reward models
    policy = LSTMPolicy(64 * 64)
    reward_model = LSTMRewardModel(64 * 64)

    # --- Initialize optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr = POLICY_LR)
    reward_optimizer = optim.Adam(reward_model.paraeters(), lr = REWARD_MODEL_LR)

    # --- Collect preference data and train reward model
    for epoch in range(N_REWARD_EPOCHS):

        # collect two segments using the policy
        seg_a = collect_segment(policy, env)
        seg_b = collect_segment(policy, env)

        # get human preference of the two sequences that were observed
        pref = get_human_preference()
        
        # update the reward model
        reward_optimizer.zero_grad()
        loss = preference_loss(reward_model, seg_a, seg_b, pref)
        loss.backward()
        reward_optimizer.step()

        # print out training update for the reward model
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Reward model loss: {loss.item():.4f}")


