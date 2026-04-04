import torch 
import torch.nn as nn
from hyperparameters import MAX_STEPS_PER_EPISODE, PPO_GAMMA, PPO_LAMBDA, PPO_EPOCHS, PPO_CLIP, VALUE_COEF, ENTROPY_COEF
import numpy as np

def preprocess_state(state):
    state = torch.from_numpy(state.astype(np.float32) / 255.0)[0].flatten().unsqueeze(0)
    return state

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
 
        next_state, _, done, info = env.step(action.item())
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
    }, info['location_stat_history']

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