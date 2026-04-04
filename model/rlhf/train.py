import os 
import sys
import copy
import time
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
from generate_preference import get_human_preference
from ppo import collect_ppo_rollout, compute_gae, ppo_update
from hyperparameters import N_ACTIONS, LEARNING_RATE, BATCH_SIZE, N_FRAMES, N_EPISODES, MAX_STEPS_PER_EPISODE, CHECKPOINT_FREQ, CHECKPOINT_DIR, START_CHECKPOINT, N_REWARD_EPOCHS, N_PREF_ITERATIONS, N_REWARD_STEPS, N_RL_STEPS, POLICY_LR, REWARD_MODEL_LR, PPO_EPOCHS, PPO_CLIP, PPO_GAMMA, PPO_LAMBDA, VALUE_COEF, ENTROPY_COEF, PPO_LR, N_PPO_ROLLOUTS, MAX_ITERATIONS, N_POLICY_LAYERS, N_VALUE_NET_LAYERS, ENV_RESET_INTERVAL, ENV_RETRY_ATTEMPTS, ENV_RETRY_DELAY
 
logger = logging.getLogger(__name__)
 
mlflow.set_experiment("RLHF Experiment 1")
 

 
# --- keep track of steps for the various quantities that we track
logging_steps = 0
ppo_rollout_steps = 0
reward_optimization_steps = 0

# --- remember the paths of all the checkpoints that were saved during this run
checkpoint_paths = []

def save_checkpoint(policy, value_net, reward_model):
    global checkpoint_paths
    global ppo_rollout_steps
    global reward_optimization_steps
    # PPO Update completed
    # Save checkpoint
    print(f"Finished PPO rollouts\nSaving Checkpoint...")
    now = datetime.now()
    run_name = mlflow.active_run().data.tags.get("mlflow.runName")
    os.makedirs(f'{CHECKPOINT_DIR}/{run_name}', exist_ok=True)
    checkpoint_path = f'{CHECKPOINT_DIR}/{run_name}/checkpoint_{now.strftime("%M")}.{now.strftime("%H")}.{now.strftime("%d")}.{now.strftime("%m")}.{now.strftime("%Y")}.pt'
    torch.save({
        "policy": policy.state_dict(),
        "value_net": value_net.state_dict(),
        "reward_model": reward_model.state_dict(),
        "ppo_rollout_steps" : ppo_rollout_steps,
        "reward_optimization_steps" :reward_optimization_steps
    }, checkpoint_path)
    checkpoint_paths.append(checkpoint_path)
    # log this checkpoint with MLFlow
    mlflow.log_text("\n".join(checkpoint_paths), "checkpoint_paths.txt")
 
def preprocess_state(state):
    state = torch.from_numpy(state.astype(np.float32) / 255.0)[0].flatten().unsqueeze(0)
    return state 
 
def rebuild_env(create_env):
    """
    Attempt to create a fresh environment, retrying up to ENV_RETRY_ATTEMPTS times
    if the environment crashes during construction. Returns the new env, or raises
    RuntimeError if all attempts fail.
    """
    for attempt in range(1, ENV_RETRY_ATTEMPTS + 1):
        try:
            logger.info(f"Building environment (attempt {attempt}/{ENV_RETRY_ATTEMPTS})...")
            env = create_env(interactive=False, realtime=False)
            env = RLHFActionWrapper(env)
            env = GrayscaleWrapper(env)
            env = FrameStackWrapper(env, N_FRAMES)
            logger.info("Environment created successfully.")
            return env
        except Exception as e:
            logger.warning(f"Environment creation attempt {attempt} failed: {e}")
            if attempt < ENV_RETRY_ATTEMPTS:
                logger.info(f"Waiting {ENV_RETRY_DELAY}s before retrying...")
                time.sleep(ENV_RETRY_DELAY)
    raise RuntimeError(f"Failed to create environment after {ENV_RETRY_ATTEMPTS} attempts.")
 
def safe_env_close(env):
    """Try to close the environment gracefully, ignoring any errors."""
    try:
        env.close()
    except Exception as e:
        logger.warning(f"Error while closing environment (ignoring): {e}")
 
def collect_segment(policy, env, render=True):
    """
    Act out a segment using the policy. Returns (segment, info) or raises
    an exception if the environment dies mid-episode.
    """
    hidden = None
    obs_list = []
 
    state = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        if render:
            env.render()
        obs = preprocess_state(state)
        logger.debug(f"obs.shape: {obs.shape}")
        with torch.no_grad():
            logits, hidden = policy(obs, hidden)
        action = torch.distributions.Categorical(logits=logits.squeeze()).sample().item()
        obs_list.append(obs.squeeze())
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
    final_info = info
 
    segment = torch.tensor(np.array(obs_list), dtype=torch.float32).unsqueeze(0)
    return segment, copy.deepcopy(final_info['location_stat_history'])
 
def reward_model_epoch(env, create_env, policy, reward_model, reward_optimizer):
    global logging_steps
    global reward_optimization_steps
 
    # --- Collect a batch of preference pairs over N_PREF_ITERATIONS ---
    batch_seg_a, batch_seg_b, batch_prefs = [], [], []
    for pref_iter in range(N_PREF_ITERATIONS):
        # --- Collect segment A, retrying on environment crash ---
        seg_a, info_a = None, None
        for attempt in range(1, ENV_RETRY_ATTEMPTS + 1):
            try:
                seg_a, info_a = collect_segment(policy, env, render=False)
                break
            except Exception as e:
                logger.warning(f"[Pref iter {pref_iter}] collect_segment A failed (attempt {attempt}): {e}")
                safe_env_close(env)
                env = rebuild_env(create_env)
        if seg_a is None:
            logger.error(f"[Pref iter {pref_iter}] Could not collect segment A after {ENV_RETRY_ATTEMPTS} attempts, skipping pref iter.")
            continue
 
        # --- Collect segment B, retrying on environment crash ---
        seg_b, info_b = None, None
        for attempt in range(1, ENV_RETRY_ATTEMPTS + 1):
            try:
                seg_b, info_b = collect_segment(policy, env, render=False)
                break
            except Exception as e:
                logger.warning(f"[Pref iter {pref_iter}] collect_segment B failed (attempt {attempt}): {e}")
                safe_env_close(env)
                env = rebuild_env(create_env)
        if seg_b is None:
            logger.error(f"[Pref iter {pref_iter}] Could not collect segment B after {ENV_RETRY_ATTEMPTS} attempts, skipping pref iter.")
            continue
 
        logger.debug(f"[Pref iter {pref_iter}] seg_a.shape: {seg_a.shape}")
 
        pref = get_human_preference(info_a, info_b)
        batch_seg_a.append(seg_a)
        batch_seg_b.append(seg_b)
        batch_prefs.append(pref)
 
    if len(batch_prefs) == 0:
        logger.error("No preference pairs collected this epoch — skipping reward model update.")
        return env   # return env so the caller always gets the current (possibly rebuilt) env
 
    # -- Convert our segments and preferences to tensors
    batch_seg_a = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch_seg_a]).squeeze(1)
    batch_seg_b = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch_seg_b]).squeeze(1)
    batch_prefs = torch.tensor(batch_prefs, dtype=torch.float32)
 
    # --- Optimize the reward model using the preference data that was just collected
    for reward_step in range(N_REWARD_STEPS):
        logger.info(f"optimizing reward model (step {reward_step})")
        reward_optimizer.zero_grad()
        logger.debug(f"batch_seg_a.shape: {batch_seg_a.shape}")
        loss = preference_loss(reward_model, batch_seg_a, batch_seg_b, batch_prefs)
        loss.backward()
        reward_optimizer.step()
 
        mlflow.log_metrics({
            "reward_model_loss": loss.item(),
        }, step=reward_optimization_steps)
        reward_optimization_steps += 1
 
    return env   # always return the (possibly rebuilt) env
 
 
def train(create_env):
    '''
    
        The main training loop for DPO model
    
    '''
    global ppo_rollout_steps
    global reward_optimization_steps
    
    # create the environment for the first time
    env = rebuild_env(create_env)
 
    # --- Initialize the policy and reward models
    policy = LSTMPolicy(64 * 64, action_dim=N_ACTIONS, num_layers=N_POLICY_LAYERS)
    reward_model = LSTMRewardModel(64 * 64, num_layers=N_POLICY_LAYERS)
    # value network for PPO. This is not necessarily part of the RLHF
    value_net = LSTMValue(64 * 64, num_layers=N_VALUE_NET_LAYERS)
 
    # --- If the checkpoint path exists, then load the checkpoint
    if os.path.exists(START_CHECKPOINT):
        checkpoint = torch.load(START_CHECKPOINT)
        policy.load_state_dict(checkpoint['policy'])
        reward_model.load_state_dict(checkpoint['reward_model'])
        value_net.load_state_dict(checkpoint['value_net'])
        ppo_rollout_steps = checkpoint['ppo_rollout_steps']
        reward_optimization_steps = checkpoint['reward_optimization_steps']
        logger.info(f"loaded checkpoint from: {START_CHECKPOINT}")
    

 
    # --- Initialize optimizers
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=REWARD_MODEL_LR)
    # Joint optimiser for policy + value net (standard PPO practice)
    ppo_optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=PPO_LR
    )
 
    
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
            "n_pref_iterations" : N_PREF_ITERATIONS,
            "n_reward_steps" : N_REWARD_STEPS
            }
        )
        # log the file that is used to generate artificial human preferences
        mlflow.log_artifact("model/rlhf/generate_preference.py")
 
        for itr in range(MAX_ITERATIONS):
            # --- Collect preference data and train reward model
            for epoch in range(N_REWARD_EPOCHS):
 
                # reward_model_epoch now returns the (possibly rebuilt) env
                try:
                    env = reward_model_epoch(
                        env,
                        create_env,
                        policy,
                        reward_model,
                        reward_optimizer
                    )
                except Exception as e:
                    logger.error(f"[Itr {itr} Epoch {epoch}] reward_model_epoch failed unexpectedly: {e}. "
                                 f"Attempting environment rebuild and continuing.")
                    safe_env_close(env)
                    env = rebuild_env(create_env)
                    continue
 
                # reset the environment every so often
                logger.debug(f"Resetting environment...")
                safe_env_close(env)
                time.sleep(5)
                env = rebuild_env(create_env)
                logger.debug(f"Created new environment!")
            
            # --- now that we have finished this, we can save a checkpoint
            save_checkpoint(
                policy, value_net, reward_model
            )
 
            # --- The parameters of the policy are fit using PPO
            print("\nStarting PPO policy optimization with learned reward model...")
 
            # do PPO rollouts using reward model
            for rollout_idx in range(N_PPO_ROLLOUTS):
            
                # reset the environment every so often
                if epoch > 0 and rollout_idx % ENV_RESET_INTERVAL == 0:
                    logger.debug(f"Resetting environment...")
                    safe_env_close(env)
                    time.sleep(5)
                    env = rebuild_env(create_env)
                    logger.debug(f"Created new environment!")
 
                logger.debug(f"PPO rollout {rollout_idx}")
 
                # 1. Collect one episode of experience — retry on env crash
                rollout, info = None, None
                for attempt in range(1, ENV_RETRY_ATTEMPTS + 1):
                    try:
                        rollout, info = collect_ppo_rollout(policy, value_net, reward_model, env)
                        break
                    except Exception as e:
                        logger.warning(f"[Rollout {rollout_idx}] collect_ppo_rollout failed "
                                       f"(attempt {attempt}): {e}")
                        safe_env_close(env)
                        env = rebuild_env(create_env)
 
                if rollout is None:
                    logger.error(f"[Rollout {rollout_idx}] Could not collect rollout after "
                                 f"{ENV_RETRY_ATTEMPTS} attempts, skipping.")
                    continue
 
                logger.debug(f"collected rollout...")
                mlflow.log_metrics(rlhf_wrapper.get_preference_metrics(info), step=ppo_rollout_steps)
 
                # 2. Run PPO update passes over that experience
                try:
                    p_loss, v_loss, entropy = ppo_update(policy, value_net, ppo_optimizer, rollout)
                except Exception as e:
                    logger.error(f"[Rollout {rollout_idx}] ppo_update failed: {e}. Skipping update.")
                    ppo_rollout_steps += 1
                    continue
 
                logger.debug(f"run ppo update...")
 
                mlflow.log_metrics({
                    "policy_loss": p_loss,
                    "value_loss": v_loss,
                    "entropy": entropy,
                }, step=ppo_rollout_steps)
                mean_reward = rollout["rewards"].mean().item()
                mlflow.log_metrics({"mean_reward": mean_reward}, step=ppo_rollout_steps)
                print(
                    f"  Rollout {rollout_idx:4d} | "
                    f"Policy loss: {p_loss:.4f} | "
                    f"Value loss: {v_loss:.4f} | "
                    f"Entropy: {entropy:.4f} | "
                    f"Mean reward: {mean_reward:.4f}"
                )
                ppo_rollout_steps += 1

            # --- Save checkpoint for the PPO rollout
            save_checkpoint(
                policy, value_net, reward_model
            )