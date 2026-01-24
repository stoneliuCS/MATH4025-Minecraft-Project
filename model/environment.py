from gym import Env, make

INTERACTIVE_PORT = 6666


def create_environment(objective: str, interactive: bool | None) -> Env:
    """
    Creates an Minecraft environment for the agent to interact with. Optionally supports an interactive mode

    """
    env = make(objective)
    if interactive:
        env.make_interactive(port=INTERACTIVE_PORT, realtime=True)
    return env
