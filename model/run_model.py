import gym
import numpy as np

def run_random_agent(env: gym.Env, num_episodes=5):
    """Run a random agent and print detailed statistics."""
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0
        episode_rewards = []
        
        # Track starting position
        if 'location' in obs:
            start_pos = obs['location']
            print(f"Starting Position: x={start_pos['x']:.2f}, y={start_pos['y']:.2f}, z={start_pos['z']:.2f}")
        
        while not done:
            # Sample random action
            action = env.action_space.sample()
            
            # In BASALT environments, sending ESC action will end the episode
            # Lets not do that
            action["ESC"] = 0
            
            # Take step
            obs, reward, done, info = env.step(action)
            env.render()
            
            step += 1
            total_reward += reward
            episode_rewards.append(reward)
            
            # Print step info every 10 steps or if significant reward
            if step % 10 == 0 or abs(reward) > 0.1:
                print(f"\nStep {step}:")
                print(f"  Reward: {reward:.4f}")
                print(f"  Total Reward: {total_reward:.4f}")
                
                if 'location' in obs:
                    pos = obs['location']
                    print(f"  Position: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
                
                # Print action taken (only movement actions since we're using RestrictedActionWrapper)
                active_actions = []
                for key, val in action.items():
                    if key in ['forward', 'back', 'left', 'right']:
                        if (isinstance(val, np.ndarray) and val.item() == 1) or val == 1:
                            active_actions.append(key)
                
                if active_actions:
                    print(f"  Actions: {', '.join(active_actions)}")
                
                # Check for special events
                if reward >= 100:
                    print(f"  🎯 GOAL REACHED! Found red wool!")
                elif reward <= -20:
                    print(f"  ⚠️  Hit yellow wool (obstacle)!")
            
            if done:
                print(f"\n{'='*60}")
                print(f"EPISODE {episode + 1} COMPLETE")
                print(f"{'='*60}")
                print(f"Total Steps: {step}")
                print(f"Total Reward: {total_reward:.4f}")
                print(f"Average Reward per Step: {total_reward/step:.4f}")
                print(f"Success: {'Yes' if total_reward >= 100 else 'No'}")
                
                # Reward breakdown
                positive_rewards = sum(r for r in episode_rewards if r > 0)
                negative_rewards = sum(r for r in episode_rewards if r < 0)
                print(f"Positive Rewards: {positive_rewards:.4f}")
                print(f"Negative Rewards: {negative_rewards:.4f}")
                
                if 'location' in obs:
                    final_pos = obs['location']
                    print(f"Final Position: x={final_pos['x']:.2f}, y={final_pos['y']:.2f}, z={final_pos['z']:.2f}")
                
                break
        
        # Small delay between episodes
        import time
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"ALL EPISODES COMPLETE")
    print(f"{'='*60}")