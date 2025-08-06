import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer


class PlayAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=500,
        batch_size=512,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=50_000)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, states):
        if isinstance(states, tuple):
            states = states[0]
        batch_size = states.shape[0]
        self.steps_done += batch_size

        actions = []
        states_tensor = states.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.policy_net(states_tensor)

        for i in range(batch_size):
            actions.append(q_values[i].argmax().item())

        return np.array(actions)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.policy_net.eval()
        self.target_net.eval()
        self.memory = ReplayBuffer(capacity=100_000)
