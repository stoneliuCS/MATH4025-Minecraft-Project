import logging
import argparse
import importlib.util
import os
import time
from minerl import *

from environment.restricted_wrapper import RestrictedActionWrapper
from .environment import create_environment
from .run_model import (
    run_random_agent,
    run_agent_with_q_learning,
    run_agent_with_learned_policy,
    evaluate_random_policy,
    save_q_table,
    load_q_table,
)
from environment.simple_environment import BoxedNavigationSimpleEnvironment

logging.basicConfig(level=logging.DEBUG)


def _load_dqn_train():
    """Import model/deep-q-learning/train.py (hyphenated dir needs importlib)."""
    path = os.path.join(os.path.dirname(__file__), "deep-q-learning", "train.py")
    spec = importlib.util.spec_from_file_location("dqn_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run():
    env = create_environment("MineRLBasaltFindCave-v0", interactive=True)
    run_random_agent(env)


def run_simple_environment():
    def pretrain():
        # 1) Baseline: random policy performance (with GUI)
        print("=" * 60)
        print("Running random baseline policy (with GUI)...")
        print("=" * 60)
        baseline_env = None
        try:
            baseline_env = create_environment(env_name, interactive=True)
            print("Waiting for Minecraft client to connect...")
            time.sleep(3)  # Give client time to connect
            random_returns = evaluate_random_policy(
                RestrictedActionWrapper(baseline_env),
                episodes=5,
                render=True,
            )
            avg_random = sum(random_returns) / len(random_returns)
            logging.info(f"Random policy average return over 5 episodes: {avg_random:.2f}")
        finally:
            if baseline_env is not None:
                baseline_env.close()
                time.sleep(2)  # Give time for cleanup
    def train():
        # 2) Train Q-learning agent (headless - no GUI for faster training)
        print("=" * 60)
        print("Training Q-learning agent (headless mode)...")
        print("=" * 60)
        train_env = None
        try:
            train_env = create_environment(env_name, interactive=False)
            q_table = run_agent_with_q_learning(RestrictedActionWrapper(train_env), render_training=False)
            # Save learned Q-table so it can be reused later without retraining
            save_q_table(q_table, "artifacts/q_table.pkl")
            return q_table
        finally:
            if train_env is not None:
                train_env.close()
                time.sleep(1)  # Give time for cleanup

    def posttrain(q_table):
        # 3) Evaluate learned policy (with GUI)
        print("=" * 60)
        print("Evaluating learned policy (with GUI)...")
        print("=" * 60)
        eval_env = None
        try:
            eval_env = create_environment(env_name, interactive=True)
            print("Waiting for Minecraft client to connect...")
            time.sleep(3)  # Give client time to connect
            learned_returns = run_agent_with_learned_policy(
                RestrictedActionWrapper(eval_env),
                q_table,
                episodes=5,
                render=True,
            )
            avg_learned = sum(learned_returns) / len(learned_returns)
            logging.info(
                f"Learned policy average return over 5 episodes: {avg_learned:.2f} "
            )
        finally:
            if eval_env is not None:
                eval_env.close()
                time.sleep(2)  # Give time for cleanup

    abs_box_env = BoxedNavigationSimpleEnvironment()
    abs_box_env.register()
    env_name = "BoxedNavigation-v0"

    q_table = train()
    posttrain(q_table)

    # 4) Final visualization of learned policy with GUI
    print("=" * 60)
    print("Final visualization of learned policy (with GUI)...")
    print("=" * 60)
    final_eval_env = None
    try:
        final_eval_env = create_environment(env_name, interactive=True)
        print("Waiting for Minecraft client to connect...")
        time.sleep(3)  # Give client time to connect
        run_agent_with_learned_policy(
            RestrictedActionWrapper(final_eval_env),
            q_table,
            episodes=1,
            render=True,
        )
    finally:
        if final_eval_env is not None:
            final_eval_env.close()


def run_learned_agent_only():
    """
    Load a previously trained Q-table from disk and run the agent in the GUI.
    Use this when you want to see the final policy again without retraining.
    """
    env_name = "BoxedNavigation-v0"
    abs_box_env = BoxedNavigationSimpleEnvironment()
    abs_box_env.register()

    # Load saved Q-table
    q_table = load_q_table("artifacts/q_table.pkl")

    # Run in interactive (GUI) mode
    print("Creating interactive environment - Minecraft window should appear...")
    eval_env = None
    try:
        eval_env = create_environment(env_name, interactive=True)
        print("Waiting for Minecraft client to connect (this may take 10-30 seconds)...")
        time.sleep(5)  # Give client time to start and connect
        print("Environment ready!")

        run_agent_with_learned_policy(
            RestrictedActionWrapper(eval_env),
            q_table,
            episodes=1,
            render=False,
        )
    finally:
        if eval_env is not None:
            eval_env.close()


def run_dqn_training():
    """Train a DQN agent using CNN on POV camera frames."""
    env_name = "BoxedNavigation-v0"
    abs_box_env = BoxedNavigationSimpleEnvironment()
    abs_box_env.register()

    dqn_train = _load_dqn_train()

    print("=" * 60)
    print("Training DQN agent (headless mode)...")
    print("=" * 60)
    train_env = None
    try:
        train_env = create_environment(env_name, interactive=False)
        dqn_train.train_dqn(RestrictedActionWrapper(train_env))
    finally:
        if train_env is not None:
            train_env.close()


def run_dqn_eval():
    """Evaluate a trained DQN agent with GUI."""
    env_name = "BoxedNavigation-v0"
    abs_box_env = BoxedNavigationSimpleEnvironment()
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
        choices=["train", "run-learned", "dqn", "dqn-eval", "sac"],
        default="train",
        help="Mode: 'train' tabular Q-learning, 'run-learned' load Q-table, 'dqn' train DQN, 'dqn-eval' evaluate DQN, 'sac' train SAC",
    )
    parser.add_argument("--render", action="store_true", help="Enable env.render() during training")
    args = parser.parse_args()

    if args.mode == "train":
        run_simple_environment()
    elif args.mode == "run-learned":
        run_learned_agent_only()
    elif args.mode == "dqn":
        run_dqn_training()
    elif args.mode == "dqn-eval":
        run_dqn_eval()
    elif args.mode == "sac":
        from model.sac.run import run as run_sac
        run_sac(render=args.render)
