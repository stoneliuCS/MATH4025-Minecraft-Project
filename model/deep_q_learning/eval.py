

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_dqn(env, model_path=CHECKPOINT_PATH, episodes=5, render=False):
    """Load a saved DQN model and run a greedy policy."""
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, N_FRAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(N_FRAMES, n_actions=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < MAX_STEPS_PER_EPISODE:
            if render:
                env.render()
            with torch.no_grad():
                t = torch.from_numpy(state.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                action = model(t).argmax(dim=1).item()
            state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        logger.info(f"Eval episode {ep}/{episodes} | Reward: {total_reward:.2f} | Steps: {step}")

    env.close()
