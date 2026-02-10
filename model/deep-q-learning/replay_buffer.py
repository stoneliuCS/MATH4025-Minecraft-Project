import numpy as np


class ReplayBuffer:
    """Circular replay buffer storing frames as uint8 for memory efficiency."""

    def __init__(self, capacity, frame_shape=(4, 64, 64)):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.states = np.empty((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *frame_shape), dtype=np.uint8)
        self.dones = np.empty(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices].astype(np.float32) / 255.0,
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices].astype(np.float32) / 255.0,
            self.dones[indices],
        )

    def __len__(self):
        return self.size
