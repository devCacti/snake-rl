import torch
import numpy as np
from dqn.agent import DQNAgent
from env.SnakeEnv import SnakeGame

GRID_SIZE = 10
CHECKPOINT_PATH = "checkpoints/dqn_snake_agent_latest.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play():
    # Create env with rendering
    env = SnakeGame(grid_size=GRID_SIZE, render_mode="human")
    obs = env.reset()
    state_dim = len(obs)
    action_dim = env.action_dim

    # Init agent and load weights
    agent = DQNAgent(state_dim, action_dim, DEVICE)
    agent.batch_size = 4096
    # "checkpoints/dqn_snake_agent_" + timestamp + ".pth"
    # Get the latest checkpoint (Based on the name)
    print(f"Loading agent from {CHECKPOINT_PATH}")
    agent.load(CHECKPOINT_PATH)
    agent.epsilon = 0  # No exploration

    state = obs
    total_reward = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

    while True:
        # Pick action from policy
        action = agent.select_action(state)[0].item()
        # select_action returns np.array

        # Step env
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        env.render()

        if done:
            print(f"Game Over! Total reward: {total_reward:.2f}")
            total_reward = 0
            state = env.reset()


if __name__ == "__main__":
    play()
