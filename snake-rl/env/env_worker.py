import multiprocessing as mp
from env.snake_game import SnakeGame


class EnvWorker(mp.Process):
    def __init__(self, conn, env_fn):
        super().__init__()
        self.conn = conn
        self.env: SnakeGame = env_fn()

    def run(self):
        while True:
            cmd, data = self.conn.recv()
            if cmd == "reset":
                obs = self.env.reset()
                self.conn.send(obs)
            elif cmd == "step":
                obs, reward, done = self.env.step(data)
                self.conn.send((obs, reward, done))
                if done:
                    obs = self.env.reset()
            elif cmd == "close":
                self.conn.close()
                break
