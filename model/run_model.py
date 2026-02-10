import gym
import numpy as np
<<<<<<< Updated upstream

def run_random_agent(env: gym.Env, num_episodes=5):
    """Run a random agent and print detailed statistics."""
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
=======
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from model.vision_model import VisionEncoder  # Import vision model

NUMBER_OF_EPISODES = 50
NUMBER_OF_STEPS = 2000
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.9

USE_VISION = True  # Set to False to compare without vision
VISION_LEARNING_RATE = 0.0001
VISION_UPDATE_FREQUENCY = 10  # Update vision every N steps


def save_q_table(q_table: dict[int, list[float]], path: str | Path) -> None:
    """
    Persist the learned Q-table to disk using pickle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(q_table, f)


def load_q_table(path: str | Path) -> dict[int, list[float]]:
    """
    Load a Q-table previously saved with save_q_table.
    """
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def run_random_agent(env: gym.Env):
    obs = env.reset()
    done = False
    breakpoint()
    while not done:
        action = env.action_space.sample()
        # In BASALT environments, sending ESC action will end the episode
        # Lets not do that
        action["ESC"] = 0
        obs, reward, done, _ = env.step(action)
        env.render()


def evaluate_random_policy(
    env: gym.Env,
    episodes: int = 5,
    render: bool = False,
):
    """
    Run a purely random policy for a few episodes and report total rewards.
    Returns a list of episode returns.
    """
    episode_returns: list[float] = []

    for episode in range(episodes):
>>>>>>> Stashed changes
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_rewards = []
        
        # Track starting position
        if 'location' in obs:
            start_pos = obs['location']
            print(f"Starting Position: x={start_pos['x']:.2f}, y={start_pos['y']:.2f}, z={start_pos['z']:.2f}")
        
        while not done:
            # Sample random action
            action = env.action_space.sample()
<<<<<<< Updated upstream
            
            # In BASALT environments, sending ESC action will end the episode
            # Lets not do that
            action["ESC"] = 0
            
            # Take step
            obs, reward, done, info = env.step(action)
            env.render()
            
=======
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()

        episode_returns.append(episode_reward)
        print(
            f"[RANDOM] Episode {episode + 1}/{episodes}, "
            f"total_reward={episode_reward:.2f}"
        )

    return episode_returns


def run_agent_with_q_learning(env: gym.Env, render_training: bool = False):
    no_valid_actions = env.action_space.n  # Should be 4
    current_epsilon = INITIAL_EPSILON
    gamma = GAMMA
    Q_table: dict[int, list[float]] = {}
    Q_update_table: dict[int, list[int]] = {}

    # NEW: Initialize vision model
    vision_model = VisionEncoder(output_dim=64) if USE_VISION else None
    vision_optimizer = optim.Adam(vision_model.parameters(), lr=VISION_LEARNING_RATE) if USE_VISION else None
    vision_buffer = []  # Store (current_pov, next_pov) pairs

    ACTION_NAMES = ["forward", "back", "left", "right"]

    # NEW: Function to extract visual features
    def extract_visual_bucket(pov, model):
        """Extract visual bucket from POV using CNN."""
        if model is None:
            return 0
        
        with torch.no_grad():
            # FIX: Make a copy to ensure positive strides
            pov_copy = np.array(pov, copy=True)
            features = model(pov_copy)
            features_np = features.detach().numpy()
            # Use top 8 features and find which is highest
            bucket = int(np.argmax(features_np[:8]))
            return bucket

    def discretize_obs(obs) -> int:
        """
        Hash the 3D position + visual features into a single integer state.
        """
        loc = obs["location_stats"]
        x = int(loc["xpos"])
        y = int(loc["ypos"])
        z = int(loc["zpos"])
        
        # NEW: Add visual information to state
        if USE_VISION and vision_model is not None:
            visual_bucket = extract_visual_bucket(obs["pov"], vision_model)
        else:
            visual_bucket = 0
        
        # Combine position and vision: state = (pos * 8) + visual_bucket
        return ((x * 1000 + y) * 1000 + z) * 8 + visual_bucket

    # NEW: Function to train vision model
    def update_vision_model(vision_buffer, model, optimizer):
        """Train vision model using temporal consistency."""
        if len(vision_buffer) < 2:
            return 0.0
        
        # Sample mini-batch
        batch_size = min(32, len(vision_buffer))
        indices = np.random.choice(len(vision_buffer), batch_size, replace=False)
        
        total_loss = 0.0
        for idx in indices:
            current_pov, next_pov = vision_buffer[idx]
            
            # Get features
            current_features = model(current_pov)
            with torch.no_grad():
                next_features = model(next_pov)
            
            # Loss: consecutive frames should have similar features
            loss = nn.MSELoss()(current_features, next_features)
            total_loss += loss.item()
        
        # Backprop
        avg_loss = total_loss / batch_size
        loss_tensor = torch.tensor(avg_loss, requires_grad=True)
        optimizer.zero_grad()
        
        # Actually need to compute loss properly
        batch_loss = 0.0
        for idx in indices:
            current_pov, next_pov = vision_buffer[idx]
            current_features = model(current_pov)
            next_features = model(next_pov).detach()
            batch_loss += nn.MSELoss()(current_features, next_features)
        
        batch_loss = batch_loss / batch_size
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        return batch_loss.item()

    def epsilon_greedy_policy(
        hashed_state, current_epsilon, Q_table: dict[int, list[float]], env: gym.Env
    ) -> int:
        """Epsilon greedy action selection."""
        rand = np.random.rand()
        if rand <= current_epsilon:
            return env.action_space.sample()
        else:
            argmax_action = max(
                range(no_valid_actions),
                key=lambda action: Q_table.setdefault(
                    hashed_state, np.array([0.0] * no_valid_actions, dtype=float)
                )[action],
            )
            return argmax_action

    def q_new_opt(
        hashed_state: int,
        hashed_state_prime: int,
        terminal_state: bool,
        reward: float,
        gamma: float,
        action: int,
        Q_table: dict[int, list[float]],
        Q_update_table: dict[int, list[int]],
    ):
        eta = 1.0 / (
            1
            + Q_update_table.setdefault(
                hashed_state, np.array([0] * no_valid_actions, dtype=int)
            )[action]
        )
        Q_update_table[hashed_state][action] = Q_update_table[hashed_state][action] + 1

        q_old = Q_table.setdefault(
            hashed_state, np.array([0.0] * no_valid_actions, dtype=float)
        )[action]

        if terminal_state:
            target = reward
        else:
            target = reward + gamma * v_old_opt(Q_table, hashed_state_prime)

        return (1 - eta) * q_old + eta * target

    def v_old_opt(Q_table, hashed_state_prime):
        max_q_old_opt = -np.inf
        for action in range(no_valid_actions):
            max_q_old_opt = max(
                max_q_old_opt,
                Q_table.setdefault(
                    hashed_state_prime, np.array([0.0] * no_valid_actions, dtype=float)
                )[action],
            )
        return max_q_old_opt

    # Training loop
    total_steps = 0
    for episode in range(NUMBER_OF_EPISODES):
        obs = env.reset()
        terminal = False
        episode_reward = 0.0
        
        for step in range(NUMBER_OF_STEPS):
            if terminal:
                print("Reached a terminal state!")
                break
            
            hashed_current_state = discretize_obs(obs)
            action_index = epsilon_greedy_policy(hashed_current_state, current_epsilon, Q_table, env)

            obs_prime, reward_prime, terminal_prime, _ = env.step(action_index)
            hashed_state_prime = discretize_obs(obs_prime)
            
            q_new_opt_val = q_new_opt(
                hashed_current_state,
                hashed_state_prime,
                terminal_prime,
                reward_prime,
                gamma,
                action_index,
                Q_table,
                Q_update_table,
            )
            Q_table[hashed_current_state][action_index] = q_new_opt_val
            
            # NEW: Update vision model
            if USE_VISION:
                vision_buffer.append((obs["pov"].copy(), obs_prime["pov"].copy()))
                
                # Keep buffer manageable
                if len(vision_buffer) > 1000:
                    vision_buffer.pop(0)
                
                # Periodically train vision
                if total_steps % VISION_UPDATE_FREQUENCY == 0 and len(vision_buffer) > 10:
                    vision_loss = update_vision_model(vision_buffer, vision_model, vision_optimizer)
                    if total_steps % 100 == 0:
                        print(f"  [Vision] loss: {vision_loss:.4f}")
            
            obs = obs_prime
            episode_reward += reward_prime
            terminal = terminal_prime
            total_steps += 1

            print(
                f"Episode {episode + 1}, Step {step + 1}: "
                f"reward={reward_prime:.3f}, "
                f"action={ACTION_NAMES[action_index]}, "
                f"episode_return={episode_reward:.3f}"
            )
            
            if render_training:
                env.render()

        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
        print(f"Episode {episode + 1}/{NUMBER_OF_EPISODES}, "
              f"reward={episode_reward:.2f}, epsilon={current_epsilon:.3f}")
    
    # NEW: Save vision model
    if USE_VISION and vision_model is not None:
        torch.save(vision_model.state_dict(), "artifacts/vision_model.pth")
        print("Saved vision model to artifacts/vision_model.pth")
    
    return Q_table

def run_agent_with_learned_policy(
    env: gym.Env,
    Q_table: dict[int, list[float]],
    episodes: int = 1,
    render: bool = True,
):
    """
    Run a greedy policy from a learned Q-table for a few episodes.
    This does not update Q-values; it only exploits what was learned.
    Returns a list of episode returns.
    """
    # Simple 4-action discrete space: 0=forward, 1=back, 2=left, 3=right
    no_valid_actions = env.action_space.n  # Should be 4
    ACTION_NAMES = ["forward", "back", "left", "right"]

    def discretize_obs(obs) -> int:
        loc = obs["location_stats"]
        x = int(loc["xpos"])
        y = int(loc["ypos"])
        z = int(loc["zpos"])
        return (x * 1000 + y) * 1000 + z

    episode_returns: list[float] = []

    for episode in range(episodes):
        obs = env.reset()
        terminal = False
        episode_reward = 0.0

        step = 0
        while not terminal and step < NUMBER_OF_STEPS:
>>>>>>> Stashed changes
            step += 1
            total_reward += reward
            episode_rewards.append(reward)
            
            # Print step info every 10 steps or if significant reward
            if step % 10 == 0 or abs(reward) > 0.1:
                print(f"\nStep {step}:")
                print(f"  Reward: {reward:.4f}")
                print(f"  Total Reward: {total_reward:.4f}")
                
                if 'location' in obs:
                    pos = obs['location']
                    print(f"  Position: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
                
                # Print action taken (only movement actions since we're using RestrictedActionWrapper)
                active_actions = []
                for key, val in action.items():
                    if key in ['forward', 'back', 'left', 'right']:
                        if (isinstance(val, np.ndarray) and val.item() == 1) or val == 1:
                            active_actions.append(key)
                
                if active_actions:
                    print(f"  Actions: {', '.join(active_actions)}")
                
                # Check for special events
                if reward >= 100:
                    print(f"  🎯 GOAL REACHED! Found red wool!")
                elif reward <= -20:
                    print(f"  ⚠️  Hit yellow wool (obstacle)!")
            
            if done:
                print(f"\n{'='*60}")
                print(f"EPISODE {episode + 1} COMPLETE")
                print(f"{'='*60}")
                print(f"Total Steps: {step}")
                print(f"Total Reward: {total_reward:.4f}")
                print(f"Average Reward per Step: {total_reward/step:.4f}")
                print(f"Success: {'Yes' if total_reward >= 100 else 'No'}")
                
                # Reward breakdown
                positive_rewards = sum(r for r in episode_rewards if r > 0)
                negative_rewards = sum(r for r in episode_rewards if r < 0)
                print(f"Positive Rewards: {positive_rewards:.4f}")
                print(f"Negative Rewards: {negative_rewards:.4f}")
                
                if 'location' in obs:
                    final_pos = obs['location']
                    print(f"Final Position: x={final_pos['x']:.2f}, y={final_pos['y']:.2f}, z={final_pos['z']:.2f}")
                
                break
        
        # Small delay between episodes
        import time
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"ALL EPISODES COMPLETE")
    print(f"{'='*60}")