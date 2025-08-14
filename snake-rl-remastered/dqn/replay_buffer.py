import random
import torch
from collections import deque
from tensor.tensor_ops import to_tensor


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                to_tensor(state, torch.float32, self.device),
                to_tensor(action, torch.int64, self.device),
                to_tensor(reward, torch.float32, self.device),
                to_tensor(next_state, torch.float32, self.device),
                to_tensor(done, torch.bool, self.device),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(self.device, non_blocking=True),
            torch.stack(actions).to(self.device, non_blocking=True),
            torch.stack(rewards).to(self.device, non_blocking=True),
            torch.stack(next_states).to(self.device, non_blocking=True),
            torch.stack(dones).to(self.device, non_blocking=True),
        )
