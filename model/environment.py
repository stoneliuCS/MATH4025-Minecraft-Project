from gym import Env, make
import logging

INTERACTIVE_PORT = 6666
_logger = logging.getLogger(__name__)


def create_environment(objective: str, interactive: bool = False) -> Env:
    """
    Creates an Minecraft environment for the agent to interact with. Optionally supports an interactive mode.

    When interactive=True, waits for the Minecraft client to connect before returning.
    """
    _logger.info(f"Creating environment: {objective} (interactive={interactive})")
    env = make(objective)
    if interactive:
        _logger.info(f"Making environment interactive on port {INTERACTIVE_PORT}...")
        env.make_interactive(port=INTERACTIVE_PORT, realtime=True)
        _logger.info(
            "Interactive mode enabled. Minecraft client should connect automatically."
        )
    return env


def make_wood_env(env_name: str, render: bool = False, interactive: bool = True, sticky_ticks: int = 5) -> Env:
    """
    Shared wrapper stack for both SAC and PPO. Returns env wrapped through ActionWrapper.
    Callers add algorithm-specific wrappers (Monitor, VecEnv, GymV21, etc.) on top.
    """
    from environment.wood_env2 import (
        ActionWrapper,
        MineBlockRewardWrapper,
        PovImageWrapper,
        RenderWrapper,
        StickyAttackWrapper,
    )
    from environment.wrappers import RobustResetWrapper

    env = create_environment(env_name, interactive=interactive)
    env = RobustResetWrapper(env, env_name=env_name)
    env = MineBlockRewardWrapper(env)
    env = StickyAttackWrapper(env, sticky_ticks=sticky_ticks)
    if render:
        env = RenderWrapper(env)
    env = PovImageWrapper(env)
    env = ActionWrapper(env)
    return env
