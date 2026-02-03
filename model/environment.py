from gym import Env, make
import time
import logging

INTERACTIVE_PORT = 6666
_logger = logging.getLogger(__name__)


def create_environment(objective: str, interactive: bool | None) -> Env:
    """
    Creates an Minecraft environment for the agent to interact with. Optionally supports an interactive mode.
    
    When interactive=True, waits for the Minecraft client to connect before returning.
    """
    _logger.info(f"Creating environment: {objective} (interactive={interactive})")
    env = make(objective)
    if interactive:
        _logger.info(f"Making environment interactive on port {INTERACTIVE_PORT}...")
        env.make_interactive(port=INTERACTIVE_PORT, realtime=True)
        _logger.info("Interactive mode enabled. Minecraft client should connect automatically.")
    return env

