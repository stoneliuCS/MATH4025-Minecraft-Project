import gym
import numpy as np
import math
import pickle
from pathlib import Path

NUMBER_OF_EPISODES = 50
NUMBER_OF_STEPS = 2000
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.9


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
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done and step < NUMBER_OF_STEPS:
            step += 1
            action = env.action_space.sample()
            # Avoid terminating via ESC if present
            if isinstance(action, dict) and "ESC" in action:
                action["ESC"] = 0
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
    no_valid_actions = math.prod(space.n for space in env.action_space.spaces.values())
    current_epsilon = INITIAL_EPSILON
    gamma = GAMMA
    Q_table: dict[int, list[float]] = {}
    Q_update_table: dict[int, list[int]] = {}

    # Build mapping between action indices and dict actions
    action_keys = list(env.action_space.spaces.keys())
    action_dims = [env.action_space.spaces[k].n for k in action_keys]

    def action_index_to_dict(index: int) -> dict:
        """Convert an integer action index to a dict action for env.step()."""
        action_dict = {}
        for i, key in enumerate(action_keys):
            action_dict[key] = index % action_dims[i]
            index //= action_dims[i]
        return action_dict

    def action_dict_to_index(action_dict: dict) -> int:
        """Convert a dict action to an integer index for Q-table."""
        index = 0
        multiplier = 1
        for i, key in enumerate(action_keys):
            index += action_dict[key] * multiplier
            multiplier *= action_dims[i]
        return index

    def discretize_obs(obs) -> int:
        """
        Hash the 3D position into a single integer state.
        This preserves uniqueness for reasonable world sizes and avoids
        collapsing many different (x, y, z) positions onto the same state.
        """
        loc = obs["location_stats"]
        x = int(loc["xpos"])
        y = int(loc["ypos"])
        z = int(loc["zpos"])
        # Assuming coordinates are not huge; tweak multipliers if needed.
        return (x * 1000 + y) * 1000 + z

    def epsilon_greedy_policy(
        hashed_state, current_epsilon, Q_table: dict[int, list[float]], env : gym.Env
    ) -> int:
        """
        From the given hashed state, returns an action INDEX based off the epsilon greedy policy.
        """
        rand = np.random.rand()
        if rand <= current_epsilon:
            sample_action = env.action_space.sample()
            # Avoid accidentally terminating the episode with ESC when exploring
            if "ESC" in sample_action:
                sample_action["ESC"] = 0
            return action_dict_to_index(sample_action)
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
    for episode in range(NUMBER_OF_EPISODES):
        obs = env.reset()
        terminal = False
        episode_reward = 0.0
        for step in range(NUMBER_OF_STEPS):
            if terminal:
                break
            hashed_current_state = discretize_obs(obs)
            action_index = epsilon_greedy_policy(hashed_current_state, current_epsilon, Q_table, env)
            action_dict = action_index_to_dict(action_index)
            obs_prime, reward_prime, terminal_prime, _ = env.step(action_dict)
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
            obs = obs_prime
            episode_reward += reward_prime
            terminal = terminal_prime
            # Print per-step reward and running episode return
            print(
                f"Episode {episode + 1}, step {step + 1}, "
                f"reward={reward_prime:.3f}, "
                f"episode_return_so_far={episode_reward:.3f}"
            )
            if render_training:
                env.render()

        # Decay epsilon after each episode, but keep a minimum level of exploration
        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
        print(f"Episode {episode + 1}/{NUMBER_OF_EPISODES}, "
              f"reward={episode_reward:.2f}, epsilon={current_epsilon:.3f}")
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
    # Rebuild action mapping for this environment
    action_keys = list(env.action_space.spaces.keys())
    action_dims = [env.action_space.spaces[k].n for k in action_keys]

    def action_index_to_dict(index: int) -> dict:
        action_dict = {}
        for i, key in enumerate(action_keys):
            action_dict[key] = index % action_dims[i]
            index //= action_dims[i]
        return action_dict

    def discretize_obs(obs) -> int:
        loc = obs["location_stats"]
        x = int(loc["xpos"])
        y = int(loc["ypos"])
        z = int(loc["zpos"])
        return (x * 1000 + y) * 1000 + z

    no_valid_actions = math.prod(space.n for space in env.action_space.spaces.values())

    episode_returns: list[float] = []

    for episode in range(episodes):
        obs = env.reset()
        terminal = False
        episode_reward = 0.0

        step = 0
        while not terminal and step < NUMBER_OF_STEPS:
            step += 1
            state = discretize_obs(obs)
            # Pure greedy action selection
            q_values = Q_table.get(
                state, np.array([0.0] * no_valid_actions, dtype=float)
            )
            action_index = int(np.argmax(q_values))
            action_dict = action_index_to_dict(action_index)

            obs, reward, terminal, _ = env.step(action_dict)
            episode_reward += reward

            if render:
                env.render()

        episode_returns.append(episode_reward)
        print(
            f"[EVAL] Episode {episode + 1}/{episodes}, "
            f"total_reward={episode_reward:.2f}"
        )

    return episode_returns
