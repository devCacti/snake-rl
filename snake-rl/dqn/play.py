import torch
import numpy as np
from typing import cast
from dqn.play_agent import PlayAgent
from env.snake_game import SnakeGame
from gym.spaces import Discrete

CHECKPOINT_PATH = "checkpoints/dqn_snake_agent.pth"
GRID_SIZE = 10


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create env with rendering
    env = SnakeGame(grid_size=GRID_SIZE, render_mode="human")
    obs, _ = env.reset()
    state_dim = len(obs)
    action_dim = int(cast(Discrete, env.action_space).n)

    # Init agent and load weights
    agent = PlayAgent(state_dim, action_dim, device)
    agent.load(CHECKPOINT_PATH)
    agent.epsilon = 0.0  # No exploration

    state = np.array(obs, dtype=np.float32)
    total_reward = 0

    while True:
        # Pick action from policy
        action = agent.select_action(
            torch.tensor([state], dtype=torch.float32).to(device)
        )[
            0
        ]  # select_action returns np.array

        # Step env
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = np.array(next_state, dtype=np.float32)

        if done:
            print(f"Game Over! Total reward: {total_reward:.2f}")
            total_reward = 0
            state, _ = env.reset()


if __name__ == "__main__":
    play()
