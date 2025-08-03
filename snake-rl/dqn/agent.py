import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=500,
        batch_size=64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=250_000)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, states):
        sample = random.random()
        if isinstance(states, tuple):
            states = states[0]
        batch_size = states.shape[0]
        self.steps_done += batch_size

        # Epsilon decay (same for whole batch)
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

        actions = []
        states_tensor = torch.FloatTensor(states).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(states_tensor)

        #! Epsilon-greedy action selection
        for i in range(batch_size):
            # * If a random chosen number is less than the epsilon threshold, it will generate random actions for all environments in the batch
            if random.random() < eps_threshold:
                actions.append(random.randint(0, self.action_dim - 1))

            # * Otherwise, it will select the action with the highest Q-value
            else:
                actions.append(q_values[i].argmax().item())

        return np.array(actions)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).to(self.device)
        with torch.no_grad():
            max_next_q_values = (
                self.target_net(next_states).max(1)[0].unsqueeze(1).to(self.device)
            )
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.policy_net.eval()
        self.target_net.eval()
        self.memory = ReplayBuffer(capacity=100_000)
