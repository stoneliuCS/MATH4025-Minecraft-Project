# Project Architecture  
- State: 4 stacked 64×64 grayscale frames from the agent's POV camera — lets the network perceive motion
- Architecture: 3 conv layers (Nature DQN style) → 512-unit FC → 4 Q-values
- Replay buffer: 100k transitions stored as uint8, converted to float32 only when sampled
- Target network: Separate copy of the CNN, hard-updated every 1000 steps to stabilize training
- Epsilon schedule: Linear decay from 1.0 → 0.05 over 50k steps
