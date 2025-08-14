import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression  # Add at the top
from datetime import datetime
from env.SnakeEnv import SnakeGame
from dqn.agent import DQNAgent

NUM_ENVS = 4
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env():
    return lambda: SnakeGame()  # Grid Size is Default to 10


def train():
    print(f"Using device: {DEVICE}")

    envs = [SnakeGame() for _ in range(NUM_ENVS)]
    obs_dim = envs[0].observation_space.shape
    n_actions = envs[0].action_dim

    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=n_actions,
        device=DEVICE,
        epsilon_start=1.0,  # Max epsilon
        epsilon_end=0.005,  # Min epsilon
        epsilon_decay=50_000,  # High value to allow for more exploration for longer
        gamma=0.99,  # Discount factor
        batch_size=BATCH_SIZE,
        lr=1e-3,
    )

    states = torch.stack([env.reset() for env in envs])
    episode_rewards = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)

    max_steps = 100_000
    target_update_freq = 1000

    avg_rewards = []
    plt.ion()  # Turn on interactive mode
    _, ax = plt.subplots()
    (line,) = ax.plot([], [])
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Reward")
    ax.set_title("Training Progress")

    for step in range(max_steps):
        actions = agent.select_action(states)
        results = (env.step(action) for env, action in zip(envs, actions))
        next_states, rewards, dones = zip(*results)

        for i in range(NUM_ENVS):
            agent.store_transition(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
            episode_rewards[i] += rewards[i]
            if dones[i]:
                envs[i].reset()
                episode_rewards[i] = 0

        states = torch.stack(next_states)

        agent.train_step()
        agent.train_step()

        if step % target_update_freq == 0:
            print("Updating Network")
            agent.update_target_network()

        if (step % (max_steps / 2) == 0 and step != 0) or step == max_steps:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            print("Saving model...")
            agent.save("checkpoints/dqn_snake_agent_" + timestamp + ".pth")
            # Save the latest model
            agent.save("checkpoints/dqn_snake_agent_latest.pth")
            print("Model saved.")

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
    # Save the plot to "training_plots/dqn_snake_agent_(timestamp).png"
    plt.savefig(f"training_plots/dqn_snake_agent_{timestamp}.png")

    plt.ioff()
    (env.close() for env in envs)


if __name__ == "__main__":
    train()
