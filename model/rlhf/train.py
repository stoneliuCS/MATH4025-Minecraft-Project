import os 
import sys
import copy
import random
import logging
from datetime import datetime

import mlflow

import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim


# allow imports from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrappers import GrayscaleWrapper, FrameStackWrapper
from lstm_policy import LSTMPolicy
from lstm_reward import LSTMRewardModel
from preference_loss import preference_loss
from rlhf_wrapper import RLHFActionWrapper, print_info
import rlhf_wrapper
from lstm_value import LSTMValue


logger = logging.getLogger(__name__)

mlflow.set_experiment("RLHF Experiment 1")

N_ACTIONS = 11
LEARNING_RATE = 0.00001
BATCH_SIZE = 32
N_FRAMES = 1
N_EPISODES = 200
MAX_STEPS_PER_EPISODE = 100
CHECKPOINT_FREQ = 5          # episodes
CHECKPOINT_DIR =   "artifacts/rlhf"
START_CHECKPOINT = "none"
N_REWARD_EPOCHS = 10
N_RL_STEPS = 100
POLICY_LR = 1e-3
REWARD_MODEL_LR = 1e-3
PPO_EPOCHS = 4
PPO_CLIP = 0.2
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_LR = 3e-4
N_PPO_ROLLOUTS = 10
MAX_ITERATIONS = 100
N_POLICY_LAYERS = 4
N_VALUE_NET_LAYERS = 2

def preprocess_state(state):
    state = torch.from_numpy(state.astype(np.float32) / 255.0)[0].flatten().unsqueeze(0)
    return state

def collect_segment(policy, env, render = True):
    # act out a segment using the policy, and allow for the human viewer to view for judgement

    # hidden state for the LSTM
    hidden = None
    # history of observations which is used by the reward model
    obs_list = []

    # restart the environment
    state = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        # render the frame for the
        if render:
            env.render()
        obs = preprocess_state(state)
        logger.debug(f"obs.shape: {obs.shape}")
        with torch.no_grad():
            logits, hidden = policy(obs, hidden)
        action = torch.distributions.Categorical(logits = logits.squeeze()).sample().item()
        obs_list.append(obs.squeeze())
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
    final_info = info
    
    segment = torch.tensor(np.array(obs_list), dtype = torch.float32).unsqueeze(0)
    return segment, copy.deepcopy(final_info)

def get_human_preference(info_a, info_b):
    print("="*30)
    print(f"FIRST SEGMENT:")
    print_info(info_a)
    print("="*30)
    print(f"SECOND SEGMENT:")
    print_info(info_b)
    print("="*30)
    #preferred_segment = input("Which segment did you prefer (1 if you preferred the first segment, 0 for the second, and 0.5 for ties)?")
    
    pitch_a = rlhf_wrapper.average_pitch(info_a)
    pitch_b = rlhf_wrapper.average_pitch(info_b)
    
    # we wish to train the model to look forward
    if abs(pitch_a) > 25 or abs(pitch_b) > 25:
        if abs(pitch_a) < abs(pitch_b):
            logger.info("First Segment Preferred (pitch)")
            return torch.tensor([1.0])
        else:
            logger.info ("Second Segment Preferred (pitch)")
            return torch.tensor([0.0])
    
    # now we consider which one travelled farther if they both have good pitch
    hdt_a = rlhf_wrapper.horizontal_distance_traveled(info_a)
    hdt_b = rlhf_wrapper.horizontal_distance_traveled(info_b)
    if hdt_a > hdt_b:
        logger.info(f"First Segment Preferred (Distance Traveled={hdt_a})")
        return torch.tensor([1.0])
    else:
        logger.info(f"Second Segment Preferred (Distance Traveled={hdt_b})")
        return torch.tensor([0.0])


def train(env):
    '''
    
        The main training loop for DPO model
    
    '''

    env = RLHFActionWrapper(env)
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, N_FRAMES)

    
    # --- Initialize the policy and reward models
    policy = LSTMPolicy(64 * 64, action_dim= N_ACTIONS, num_layers=N_POLICY_LAYERS)
    reward_model = LSTMRewardModel(64 * 64, num_layers=N_POLICY_LAYERS)
    # value network for PPO
    value_net = LSTMValue(64 * 64, num_layers=N_VALUE_NET_LAYERS)

    # --- If the checkpoint path exists, then load the checkpoint
    if os.path.exists(START_CHECKPOINT):
        checkpoint = torch.load(START_CHECKPOINT)
        policy.load_state_dict(checkpoint['policy'])
        reward_model.load_state_dict(checkpoint['reward_model'])
        value_net.load_state_dict(checkpoint['value_net'])
        logger.info(f"loaded checkpoint from: {START_CHECKPOINT}")

    # --- Initialize optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr = POLICY_LR)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr = REWARD_MODEL_LR)

    
    with mlflow.start_run():
        mlflow.log_params(
            {
            "n_actions": N_ACTIONS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "n_frames": N_FRAMES,
            "n_episodes": N_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "checkpoint_dir": CHECKPOINT_DIR,
            "start_checkpoint" : START_CHECKPOINT,
            "n_rl_steps": N_RL_STEPS,
            "policy_lr": POLICY_LR,
            "reward_model_lr": REWARD_MODEL_LR,
            "ppo_epochs": PPO_EPOCHS,
            "ppo_clip": PPO_CLIP,
            "ppo_gamma": PPO_GAMMA,
            "ppo_lambda": PPO_LAMBDA,
            "value_coef": VALUE_COEF,
            "entropy_coef": ENTROPY_COEF,
            "ppo_lr": PPO_LR,
            "n_ppo_rollouts": N_PPO_ROLLOUTS,
            "max_iterations": MAX_ITERATIONS,
            "n_policy_layers" : N_POLICY_LAYERS,
            "n_value_net_layers" : N_VALUE_NET_LAYERS,
            }
        )
        # keep track of steps for the various quantities that we track
        logging_steps = 0

        for itr in range(MAX_ITERATIONS):
            # --- Collect preference data and train reward model
            for epoch in range(N_REWARD_EPOCHS):

                # collect two segments using the policy
                seg_a, info_a = collect_segment(policy, env)
                seg_b, info_b = collect_segment(policy, env)
                logger.debug(f"seg_a.shape: {seg_a.shape}")

                # get human preference of the two sequences that were observed
                pref = get_human_preference(info_a, info_b)
                
                # update the reward model
                reward_optimizer.zero_grad()
                loss = preference_loss(reward_model, seg_a, seg_b, pref)
                loss.backward()
                reward_optimizer.step()

                mlflow.log_metrics({
                    "preference_loss" : loss.item(),
                    "average_pitch" : rlhf_wrapper.average_pitch(info_a),
                    "horizontal distance traveled" : rlhf_wrapper.horizontal_distance_traveled(info_a)
                }, step = logging_steps)
                logging_steps += 1

                # print out training update for the reward model
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch:3d} | Reward model loss: {loss.item():.4f}")

            # --- The parameters of the policy are fit using PPO
            print("\nStarting PPO policy optimisation with learned reward model...")
        
            # Joint optimiser for policy + value net (standard PPO practice)
            ppo_optimizer = optim.Adam(
                list(policy.parameters()) + list(value_net.parameters()),
                lr=PPO_LR
            )

            # do PPO rollouts using reward model
            for rollout_idx in range(N_PPO_ROLLOUTS):
                logger.debug(f"PPO rollout {rollout_idx}")
        
                # 1. Collect one episode of experience using the current policy
                rollout = collect_ppo_rollout(policy, value_net, reward_model, env)
                logger.debug(f"collected rollout...")
        
                # 2. Run PPO update passes over that experience
                p_loss, v_loss, entropy = ppo_update(policy, value_net, ppo_optimizer, rollout)
                logger.debug(f"run ppo update...")

                mlflow.log_metrics({
                    "policy_loss" : p_loss,
                    "value_loss" : v_loss, 
                    "entropy" : entropy, 
                }, step = logging_steps)
                logging_steps += 1
                if rollout_idx % 20 == 0:
                    mean_reward = rollout["rewards"].mean().item()
                    mlflow.log_metrics({"mean_reward" : mean_reward}, step = logging_steps)
                    print(
                        f"  Rollout {rollout_idx:4d} | "
                        f"Policy loss: {p_loss:.4f} | "
                        f"Value loss: {v_loss:.4f} | "
                        f"Entropy: {entropy:.4f} | "
                        f"Mean reward: {mean_reward:.4f}"
                    )

            # PPO Update completed
            # Save checkpoint
            print(f"Finished PPO rollouts\nSaving Checkpoint...")
            now = datetime.now()
            torch.save({
                "policy" : policy.state_dict(),
                "value_net" : value_net.state_dict(), 
                "reward_model" : reward_model.state_dict()
            }, f'{CHECKPOINT_DIR}/checkpoint_{now.strftime("%H%M")}_{now.strftime("%d")}_{now.strftime("%m")}_{now.strftime("%Y")}.pt'
            )


# -----------------------------------------------------------
#   CODE FOR PPO
# -----------------------------------------------------------
def collect_ppo_rollout(policy, value_net, reward_model, env):
    """
    Run one full episode with the current policy, recording everything PPO needs:
      - observations, actions, log-probs (for the surrogate ratio)
      - values from the critic (for advantage estimation)
      - rewards from the learned reward model
    Returns a dict of tensors, all shape (T,) or (T, obs_dim).
    """
    obs_list, action_list, logprob_list, value_list, reward_list, done_list = [], [], [], [], [], []
 
    policy_hidden = None
    value_hidden  = None
 
    state = env.reset()
    for _ in range(MAX_STEPS_PER_EPISODE):
        obs = preprocess_state(state)                              # (1, obs_dim)
        obs_in = obs.unsqueeze(0)                                  # (1, 1, obs_dim) for LSTM
 
        with torch.no_grad():
            logits, policy_hidden = policy(obs_in, policy_hidden)
            value, value_hidden   = value_net(obs_in, value_hidden)
 
        dist   = torch.distributions.Categorical(logits=logits.squeeze())
        action = dist.sample()
        logprob = dist.log_prob(action)
 
        obs_list.append(obs)
        action_list.append(action)
        logprob_list.append(logprob)
        value_list.append(value.squeeze())
 
        next_state, _, done, _ = env.step(action.item())
        done_list.append(done)
        state = next_state
        if done:
            break
 
    # --- Score the whole trajectory with the learned reward model
    # segment shape: (1, T, obs_dim)
    segment = torch.cat(obs_list, dim=0).unsqueeze(0)
    T = segment.shape[1]
    with torch.no_grad():
        total_reward = reward_model(segment)   # scalar for whole segment
 
    # Distribute reward evenly across timesteps (simple credit assignment)
    per_step_reward = (total_reward / T).expand(T)
    reward_list = per_step_reward
 
    return {
        "obs":      torch.cat(obs_list, dim=0),          # (T, obs_dim)
        "actions":  torch.stack(action_list),             # (T,)
        "logprobs": torch.stack(logprob_list),            # (T,)
        "values":   torch.stack(value_list),              # (T,)
        "rewards":  reward_list,                          # (T,)
        "dones":    torch.tensor(done_list, dtype=torch.float32),  # (T,)
    }

def compute_gae(rewards, values, dones, gamma=PPO_GAMMA, lam=PPO_LAMBDA):
    """
    Generalised Advantage Estimation (Schulman et al. 2015).
    Returns advantages and discounted returns, both shape (T,).
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae   = 0.0
 
    # Bootstrap from the last value if episode didn't terminate
    next_value = values[-1] * (1.0 - dones[-1])
 
    for t in reversed(range(T)):
        next_val   = values[t + 1] if t + 1 < T else next_value
        delta      = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        last_gae   = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
 
    returns = advantages + values
    return advantages, returns

def ppo_update(policy, value_net, optimizer, rollout):
    """
    Run PPO_EPOCHS passes over the collected rollout, updating policy and value net.
    """
    obs      = rollout["obs"]        # (T, obs_dim)
    actions  = rollout["actions"]    # (T,)
    old_lps  = rollout["logprobs"].detach()
    values   = rollout["values"].detach()
    rewards  = rollout["rewards"]
    dones    = rollout["dones"]
 
    advantages, returns = compute_gae(rewards, values, dones)
    # Normalise advantages for training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
    T = obs.shape[0]
 
    for _ in range(PPO_EPOCHS):
        # Re-evaluate actions under the current policy
        # Feed the whole sequence at once (batch=1, T, obs_dim)
        logits, _ = policy(obs.unsqueeze(0))          # (1, T, n_actions)
        logits     = logits.squeeze(0)                # (T, n_actions)
        dist       = torch.distributions.Categorical(logits=logits)
        new_lps    = dist.log_prob(actions)           # (T,)
        entropy    = dist.entropy().mean()
 
        # Clipped surrogate objective
        ratio      = torch.exp(new_lps - old_lps)
        surr1      = ratio * advantages
        surr2      = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
 
        # Value loss
        new_values, _ = value_net(obs.unsqueeze(0))   # (1, T, 1)
        new_values     = new_values.squeeze()          # (T,)
        value_loss     = nn.functional.mse_loss(new_values, returns)
 
        # Combined loss
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
 
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value_net.parameters()), max_norm=0.5
        )
        optimizer.step()
 
    return policy_loss.item(), value_loss.item(), entropy.item()