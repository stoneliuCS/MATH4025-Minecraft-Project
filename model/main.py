import logging
from minerl import *
from .environment import create_environment
from .run_model import run_random_agent
from environment.simple_environment import BoxedNavigationSimpleEnvironment

logging.basicConfig(level=logging.DEBUG)


def run():
    env = create_environment("MineRLBasaltFindCave-v0", interactive=True)
    run_random_agent(env)

def run_simple_environment():
    abs_box_env = BoxedNavigationSimpleEnvironment()
    abs_box_env.register()
    env_name = "BoxedNavigation-v0"
    env = create_environment(env_name, interactive=True)
    run_random_agent(env)


if __name__ == "__main__":
    run_simple_environment()
