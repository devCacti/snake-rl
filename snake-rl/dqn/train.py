import numpy as np
import torch
from env.snake_game import SnakeGame
from dqn.agent import DQNAgent
import gym

NUM_ENVS = 25
BATCH_SIZE = 128


def make_env():
    return lambda: SnakeGame(grid_size=10)


def train():
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])  # type: ignore

    obs_dim = envs.single_observation_space.shape[0]  # type: ignore
    n_actions = envs.single_action_space.n  # type: ignore

    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=n_actions,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,  # Epsilon decay steps, 500 means it decays over 500 steps
        gamma=0.99,  # Discount factor
        lr=1e-3,  # Learning rate, it means the optimizer will update the model weights with this learning rate
        batch_size=BATCH_SIZE,
    )

    print(f"Using device: {agent.device}")

    states, infos = envs.reset()
    episode_rewards = np.zeros(NUM_ENVS)
    envs.render_mode = "human"

    max_steps = 200_000
    target_update_freq = 1000

    for step in range(max_steps):
        actions = agent.select_action(states)
        next_states, rewards, terminations, truncations, infos = envs.step(actions)  # type: ignore
        dones = np.logical_or(terminations, truncations)

        for i in range(NUM_ENVS):
            agent.store_transition(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
            episode_rewards[i] += rewards[i]
            if dones[i]:
                episode_rewards[i] = 0

        agent.train_step()

        states = next_states

        if step % target_update_freq == 0:
            agent.update_target_network()

    envs.close()


if __name__ == "__main__":
    train()
