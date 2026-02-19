import torch
from .ppo import PPOAgent
from .wrappers import GrayscaleWrapper, FrameStackWrapper


def train_ppo(env,
              total_steps=1_000_000,
              rollout_length=2048):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, 4)

    agent = PPOAgent(n_actions=4, device=device)

    state = env.reset()
    episode_reward = 0
    global_step = 0

    while global_step < total_steps:
        print("Step:", global_step)


        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        # ---------------------------------------
        # Collect rollout
        # ---------------------------------------
        for _ in range(rollout_length):

            action, log_prob, value = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            states.append(
                torch.from_numpy(state.astype("float32") / 255.0)
            )
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value.item())

            state = next_state
            episode_reward += reward
            global_step += 1

            if done:
                print(f"Episode reward: {episode_reward:.2f}")
                state = env.reset()
                episode_reward = 0

        # ---------------------------------------
        # Compute GAE
        # ---------------------------------------
        with torch.no_grad():
            next_value = agent.model(
                torch.from_numpy(state.astype("float32") / 255.0)
                .unsqueeze(0)
                .to(device)
            )[1].item()

        advantages, returns = agent.compute_gae(
            rewards, values, dones, next_value
        )

        # ---------------------------------------
        # PPO Update
        # ---------------------------------------
        agent.update(states, actions, log_probs, returns, advantages)

    torch.save(agent.model.state_dict(), "artifacts/ppo_model.pt")
    print("Training complete.")
