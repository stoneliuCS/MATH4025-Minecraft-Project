import logging
import minerl
from environment import create_environment
from run_model import run_random_agent

logging.basicConfig(level=logging.DEBUG)


def run():
    env = create_environment("MineRLBasaltFindCave-v0", interactive=False)
    run_random_agent(env)


if __name__ == "__main__":
    run()
