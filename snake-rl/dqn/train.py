import numpy as np
import torch
from env.snake_game import SnakeGame
from dqn.agent import DQNAgent
import gym
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import pandas as pd
from sklearn.linear_model import LinearRegression  # Add at the top

NUM_ENVS = 25
BATCH_SIZE = 256


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
        epsilon_start=1.0,  # Max epsilon
        epsilon_end=0.005,  # Min epsilon
        epsilon_decay=15000,  # High value to allow for more exploration for longer
        gamma=0.99,  # Discount factor
        batch_size=BATCH_SIZE,
    )

    print(f"Using device: {agent.device}")

    states, infos = envs.reset()
    episode_rewards = np.zeros(NUM_ENVS)
    envs.render_mode = "training"

    max_steps = 17_500  # 20k steps is already a lot, this can train an entire agent in a few minutes
    target_update_freq = 1000

    avg_rewards = []
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Reward")
    ax.set_title("Training Progress")

    for step in range(max_steps):
        actions = agent.select_action(states)
        next_states, rewards, terminations, truncations, infos = envs.step(actions)  # type: ignore
        dones = np.logical_or(terminations, truncations)

        if step % agent.epsilon_decay == 0:
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.gamma)

        for i in range(NUM_ENVS):
            agent.store_transition(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
            episode_rewards[i] += rewards[i]
            if dones[i]:
                episode_rewards[i] = 0

        # Train the agent multiple times per step
        # This is to ensure that the agent learns from the transitions
        # This is a common practice in DQN to stabilize training
        agent.train_step()
        agent.train_step()
        agent.train_step()
        agent.train_step()
        agent.train_step()

        if step % target_update_freq == 0:
            avg_reward = np.mean(episode_rewards)
            avg_rewards.append(avg_reward)
            print(f"Step: {step}, Average Reward: {avg_reward:.2f}")

            # Update main line
            x_vals = np.arange(len(avg_rewards)) * target_update_freq
            line.set_xdata(x_vals)
            line.set_ydata(avg_rewards)

            # --- Moving average trend line (window=10) ---
            if len(avg_rewards) >= 2:
                ma = pd.Series(avg_rewards).rolling(window=5).mean()
                if not hasattr(ax, "ma_line"):
                    (ax.ma_line,) = ax.plot(
                        x_vals, ma, color="red", linewidth=2, label="Moving Avg"
                    )
                else:
                    ax.ma_line.set_xdata(x_vals)
                    ax.ma_line.set_ydata(ma)
            else:
                if hasattr(ax, "ma_line"):
                    ax.ma_line.set_xdata([])
                    ax.ma_line.set_ydata([])

            # --- Linear regression trend line (every 1000 steps) ---
            if len(avg_rewards) > 1:
                x = x_vals.reshape(-1, 1)
                y = np.array(avg_rewards)
                model = LinearRegression().fit(x, y)
                y_pred = model.predict(x)
                if not hasattr(ax, "trend_line"):
                    (ax.trend_line,) = ax.plot(
                        x_vals,
                        y_pred,
                        color="green",
                        linestyle="--",
                        linewidth=2,
                        label="Trend",
                    )
                else:
                    ax.trend_line.set_xdata(x_vals)
                    ax.trend_line.set_ydata(y_pred)

            # Only add legend once
            if not hasattr(ax, "_legend_added"):
                ax.legend()
                ax._legend_added = True

            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)

        states = next_states

        if step % target_update_freq == 0:
            agent.update_target_network()

    # Get the date and time for the checkpoint's filename
    from datetime import datetime

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    print("Training complete. Saving model...")
    agent.save("checkpoints/dqn_snake_agent_" + timestamp + ".pth")
    agent.save("checkpoints/dqn_snake_agent_latest.pth")  # Save the latest model
    print("Model saved.")

    # Save the plot to "training_plots/dqn_snake_agent_(timestamp).png"
    plt.savefig(f"training_plots/dqn_snake_agent_{timestamp}.png")
    plt.ioff()  # Turn off interactive mode
    envs.close()


if __name__ == "__main__":
    train()
