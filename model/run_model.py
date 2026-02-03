import gym
import numpy as np
import math

NUMBER_OF_EPISODES = 100
NUMBER_OF_STEPS = 100000
DECAY_RATE = 0.99995
GAMMA = 0.9


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



def run_agent_with_q_learning(env: gym.Env):
    no_valid_actions = math.prod(space.n for space in env.action_space.spaces.values())
    current_epsilon = DECAY_RATE
    gamma = GAMMA
    Q_table: dict[int, list[float]] = {}
    Q_update_table: dict[int, list[int]] = {}
    total_reward = 0

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
      loc = obs['location_stats']
      return int(loc['xpos']) + int(loc['ypos']) +  int(loc['zpos'])

    def epsilon_greedy_policy(
        hashed_state, current_epsilon, Q_table: dict[int, list[float]], env : gym.Env
    ) -> int:
        """
        From the given hashed state, returns an action INDEX based off the epsilon greedy policy.
        """
        rand = np.random.rand()
        if rand <= current_epsilon:
            sample_action = env.action_space.sample()
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
    for _ in range(NUMBER_OF_EPISODES):
        obs = env.reset()
        terminal = False
        for _ in range(NUMBER_OF_STEPS):
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
                            terminal,
                            reward_prime,
                            gamma,
                            action_index,
                            Q_table,
                            Q_update_table,
                        )
            Q_table[hashed_current_state][action_index] = q_new_opt_val
            obs = obs_prime
            total_reward += reward_prime
            terminal = terminal_prime
            env.render()
    return Q_table
