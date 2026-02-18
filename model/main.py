import logging
import argparse
import importlib.util
import os
import time
from minerl import *

#from environment.restricted_wrapper import RestrictedActionWrapper
#from environment.distance_wrapper import DistanceActionWrapper




from .environment import create_environment
from environment.simple_environment import BoxedNavigationSimpleEnvironment
from environment.world_2 import World2Environment

logging.basicConfig(level=logging.DEBUG)


def _load_dqn_train():
    """Import model/deep_q_learning/train.py (hyphenated dir needs importlib)."""
    path = os.path.join(os.path.dirname(__file__), "deep_q_learning", "train.py")
    spec = importlib.util.spec_from_file_location("dqn_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_dqn_training():
    """Train a DQN agent using CNN on POV camera frames."""

    # register our training environment 
    env_name = "World2"
    abs_box_env = World2Environment()
    abs_box_env.register()

    # load the code for training the DQN
    dqn_train = _load_dqn_train()

    print("=" * 60)
    print("Training DQN agent (headless mode)...")
    print("=" * 60)


    train_env = None
    try:
        train_env = create_environment(env_name, interactive=True)
        
        dqn_train.train_dqn(train_env)
    finally:
        if train_env is not None:
            train_env.close()


def run_dqn_eval():
    """Evaluate a trained DQN agent with GUI."""
    env_name = "World2"
    abs_box_env = World2Environment()
    abs_box_env.register()

    dqn_train = _load_dqn_train()

    print("=" * 60)
    print("Evaluating DQN agent (with GUI)...")
    print("=" * 60)
    eval_env = None
    try:
        eval_env = create_environment(env_name, interactive=True)
        print("Waiting for Minecraft client to connect...")
        time.sleep(5)
        dqn_train.evaluate_dqn(RestrictedActionWrapper(eval_env), render=True)
    finally:
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-learning agent training or evaluation")
    parser.add_argument(
        "--mode",
        choices=["train", "run-learned", "dqn", "dqn-eval"],
        default="train",
        help="Mode: 'train' tabular Q-learning, 'run-learned' load Q-table, 'dqn' train DQN, 'dqn-eval' evaluate DQN",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_simple_environment()
    elif args.mode == "run-learned":
        run_learned_agent_only()
    elif args.mode == "dqn":
        run_dqn_training()
    elif args.mode == "dqn-eval":
        run_dqn_eval()
