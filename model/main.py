import logging
from minerl import *
from environment import create_environment
from run_model import run_random_agent

logging.basicConfig(level=logging.DEBUG)


def run():
    env = create_environment("MineRLBasaltCreateVillageAnimalPen-v0", interactive=True)
    run_random_agent(env)


if __name__ == "__main__":
    run()
