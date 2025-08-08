import torch
import multiprocessing as mp

from env.env_worker import EnvWorker
from env.snake_game import SnakeGame
from tensor.to_tensor import to_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParallelEnvManager:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.conns, self.workers = [], []

        # Create a temporary env to extract specs
        temp_env: SnakeGame = env_fns[0]()

        self.single_observation_space = temp_env.observation_space.shape
        self.single_action_space = temp_env.action_dim

        del temp_env

        for env_fn in env_fns:
            parent_conn, child_conn = mp.Pipe()
            worker = EnvWorker(child_conn, env_fn)
            worker.start()
            self.conns.append(parent_conn)
            self.workers.append(worker)

    def reset(self):
        for conn in self.conns:
            conn.send(("reset", None))
        results = [conn.recv() for conn in self.conns]  # list of obs from each env
        obs = torch.stack([to_tensor(o, torch.float32, DEVICE) for o in results])
        print("Reset")
        return obs

    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn.send(("step", action.item()))
        results = [conn.recv() for conn in self.conns]
        obs, rewards, dones = zip(*results)
        obs = torch.stack([to_tensor(o, torch.float32, DEVICE) for o in obs])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)
        return obs, rewards, dones

    def close(self):
        for conn in self.conns:
            conn.send(("close", None))
        for worker in self.workers:
            worker.join()
