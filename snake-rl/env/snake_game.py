# Imports

import numpy as np
import gym
from gym import spaces
import random


class SnakeGame(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeGame, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Start moving right
        self.food = self._place_food()
        return self._get_observation()

    def step(self, action):
        # Action Space
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:  # Up
            self.direction = (-1, 0)
        elif action == 1:  # Down
            self.direction = (1, 0)
        elif action == 2:  # Left
            self.direction = (0, -1)
        elif action == 3:  # Right
            self.direction = (0, 1)

        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )

        if new_head in self.snake or not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        ):
            return self._get_observation(), -10, True, {}  # Game over

        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self._place_food()
            reward = 10
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = -1

        return self._get_observation(), reward, False, {}

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for segment in self.snake:
            obs[segment] = 1.0
        obs[self.food] = -1.0
        return obs

    def _place_food(self):
        while True:
            food_position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if food_position not in self.snake:
                return food_position
