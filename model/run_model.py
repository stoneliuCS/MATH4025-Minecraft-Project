import gym


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
