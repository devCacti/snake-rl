import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression  # Add at the top
from datetime import datetime
from env.snake_game import SnakeGame
from dqn.agent import DQNAgent
from env.parallel_env_manager import ParallelEnvManager

NUM_ENVS = 6  # I only have 6 logical processors
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_BUFFER_SIZE = 10_000


def make_env():
    return lambda: SnakeGame()  # Grid Size is Default to 10


def train():

    envs = ParallelEnvManager([make_env() for _ in range(NUM_ENVS)])

    obs_dim = envs.single_observation_space  # type: ignore
    n_actions = envs.single_action_space  # type: ignore

    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=n_actions,
        device=DEVICE,
        epsilon_start=1.0,  # Max epsilon
        epsilon_end=0.005,  # Min epsilon
        epsilon_decay=25000,  # High value to allow for more exploration for longer
        gamma=0.99,  # Discount factor
        batch_size=BATCH_SIZE,
        lr=5e-4,
    )

    print(f"Using device: {agent.device}")

    states = envs.reset()
    episode_rewards = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)

    max_steps = 200_000
    target_update_freq = 500

    avg_rewards = []
    plt.ion()  # Turn on interactive mode
    _, ax = plt.subplots()
    (line,) = ax.plot([], [])
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Reward")
    ax.set_title("Training Progress")

    for step in range(max_steps):
        actions = agent.select_action(states)
        next_states, rewards, dones = envs.step(actions)

        for i in range(NUM_ENVS):
            agent.store_transition(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
            episode_rewards[i] += rewards[i]
            if dones[i]:
                episode_rewards[i] = 0

        states = next_states

        agent.train_step()

        if step % target_update_freq == 0:
            agent.soft_update()

        if step % (max_steps / 4) == 0 and step != 0:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            print("Saving model...")
            agent.save("checkpoints/dqn_snake_agent_" + timestamp + ".pth")
            agent.save(
                "checkpoints/dqn_snake_agent_latest.pth"
            )  # Save the latest model
            print("Model saved.")

        # Plot Related
        if step % (target_update_freq * 4) == 0:

            import numpy

            avg_reward = torch.mean(episode_rewards).item()
            avg_rewards.append(avg_reward)
            print(f"Step: {step}, Average Reward: {avg_reward:.2f}")

            # Update main line
            x_vals = numpy.arange(len(avg_rewards)) * target_update_freq * 4
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
                y = numpy.array(avg_rewards)
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

    # Get the date and time for the checkpoint's filename
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
