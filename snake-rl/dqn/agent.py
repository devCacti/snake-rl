import math
import torch
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer
from tensor.to_tensor import to_tensor


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=15000,
        batch_size=256,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=200_000, device=device)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, states):
        # Handle tuple input (e.g., from some Gym wrappers)
        if isinstance(states, tuple):
            states = states[0]

        # Convert to tensor
        states_tensor = to_tensor(states, dtype=torch.float32, device=self.device)

        # If single state, unsqueeze to make it batch-like
        if states_tensor.dim() == 1:
            states_tensor = states_tensor.unsqueeze(0)

        batch_size = states_tensor.shape[0]
        self.steps_done += batch_size

        # Epsilon decay
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

        with torch.no_grad():
            q_values = self.policy_net(states_tensor)

        # Vectorized epsilon-greedy
        random_values = torch.rand(batch_size, device=self.device)
        greedy_actions = q_values.argmax(dim=1)
        random_actions = torch.randint(
            0, self.action_dim, (batch_size,), device=self.device
        )

        actions = torch.where(
            random_values < eps_threshold, random_actions, greedy_actions
        )

        return actions

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.float().unsqueeze(1)  # Convert bool to float for math

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = torch.nn.SmoothL1Loss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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
        self.memory = ReplayBuffer(capacity=100_000, device=self.device)
