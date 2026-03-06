from basalt_utils.wrappers import FrameSkip

from environment.wood_environment import GatherWoodEnvironment

REPEAT_ACTION = 4


def build_sac_environment():
    env = GatherWoodEnvironment()
    # Repeats the action
    env = FrameSkip(env=env, n_repeat=REPEAT_ACTION)
