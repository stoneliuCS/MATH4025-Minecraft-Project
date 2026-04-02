from gym import Env, make
import time
import logging
from environment.world import WorldEnvironment

INTERACTIVE_PORT = 6666
_logger = logging.getLogger(__name__)


def create_environment(objective: str, interactive: bool, realtime: bool) -> Env:
    """
    Creates an Minecraft environment for the agent to interact with. Optionally supports an interactive mode.
    
    When interactive=True, waits for the Minecraft client to connect before returning.
    """
    _logger.info(f"Creating environment: {objective} (interactive={interactive})")
    env = make(objective)
    if interactive:
        _logger.info(f"Making environment interactive on port {INTERACTIVE_PORT}...")
        env.make_interactive(port=INTERACTIVE_PORT, realtime=realtime)
        _logger.info("Interactive mode enabled. Minecraft client should connect automatically.")
    return env

def create_environment_rlhf(interactive = True, realtime = False):
    env_name = "NewWorld"
    abs_box_env = WorldEnvironment(
        world_path = "/Users/cjryan/Desktop/MATH4025-Minecraft-Project/environment/worlds/rl_test_world_6.zip"
    )
    abs_box_env.register()
    train_env = create_environment(env_name, interactive=interactive, realtime=realtime)
    return train_env