import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                np.int64(action),
                np.float32(reward),
                np.array(next_state, dtype=np.float32),
                np.bool_(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.bool_),
        )
