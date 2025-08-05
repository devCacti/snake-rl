import numpy as np
import gym
from gym import spaces
import random
from torch import _euclidean_dist

# Rewards
EAT_FOOD = +1.0
DIE = -1.0
STEP_PENALTY = -0.01
ALIVE = +0.02

# Reward scaling factors
ALIGN_SCALE = 0.05
DIST_SCALE = 0.1
DANGER_SCALE = 0.05

# Constants for rendering
CELL_SIZE = 20  # Size of each cell in pixels


class SnakeGame(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode="training"):
        super(SnakeGame, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.snake = [(grid_size // 2, grid_size // 2)]
        self.direction = (0, 1)  # Start moving right
        self.food = self._place_food()
        self.render_mode = render_mode
        self.reset()
        obs = self._get_observation()  # Initial observation
        self.step_count = 0
        self.observation_space = (
            spaces.Box(  # Calculates the size of the observation space
                low=-1.0, high=1.0, shape=(len(obs),), dtype=np.float32
            )
        )

    def reset(self, *, seed=None, options=None):
        # Do NOT call super().reset(seed=seed)
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)
        self.food = self._place_food()
        obs = self._get_observation()

        # Or return obs, {} is for compatibility with gym API, it also allows for additional info if needed
        return (obs, {})

    def step(self, action):
        self.step_count += 1

        # Action mapping: 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:  # Up
            self.direction = (-1, 0)
        elif action == 1:  # Down
            self.direction = (1, 0)
        elif action == 2:  # Left
            self.direction = (0, -1)
        elif action == 3:  # Right
            self.direction = (0, 1)

        # Calculates the new head position based on the current direction
        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )

        self.render(self.render_mode)

        # Calculates the delta distance to the food
        # This is used to determine if the snake is getting closer or further from the food
        prev_dist = self._euclidean_dist(self.snake[0], self.food)
        new_dist = self._euclidean_dist(new_head, self.food)

        if new_head in self.snake or not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        ):
            return self._get_observation(), DIE, True, False, {}  # Game over

        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self._place_food()
            reward = EAT_FOOD  # Reward for eating food
        else:
            self.snake.insert(0, new_head)  # Always move the head
            self.snake.pop()  # Remove the tail if not eating

            distance_delta = prev_dist - new_dist
            reward = DIST_SCALE * distance_delta  # Reward for getting closer to food

            alignment = np.array(
                [
                    self.food[0] - new_head[0] / self.grid_size,
                    self.food[1] - new_head[1] / self.grid_size,
                ]
            )

            dirv = np.array(self.direction)

            reward += ALIGN_SCALE * np.dot(
                alignment, dirv
            )  # Reward for aligning with food

            num_dangers = sum(
                self.is_danger(new_head, dx, dy)
                for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]
            )

            reward -= DANGER_SCALE * num_dangers  # Penalty for danger
            reward += STEP_PENALTY  # Small penalty for each step taken
            reward += ALIGN_SCALE * (self.direction[0] + self.direction[1])
            reward += ALIVE  # Small positive reward for moving

        # c_step += 1
        return self._get_observation(), reward, False, False, {}

    def _euclidean_dist(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def is_danger(self, from_pos, offset_x, offset_y):
        y, x = from_pos
        next_x = x + offset_x
        next_y = y + offset_y
        return (
            not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size)
            or (next_y, next_x) in self.snake
        )

    def _get_observation(self):
        head = self.snake[0]
        head_y, head_x = head
        food_y, food_x = self.food

        rel_x = (food_x - head_x) / self.grid_size
        rel_y = (food_y - head_y) / self.grid_size

        dx, dy = self.direction

        # One-hot direction
        dir_up = int((dx, dy) == (-1, 0))
        dir_down = int((dx, dy) == (1, 0))
        dir_left = int((dx, dy) == (0, -1))
        dir_right = int((dx, dy) == (0, 1))

        # Danger detection
        danger_front = self.is_danger(head, dx, dy)
        danger_left = self.is_danger(head, -dy, dx)  # 90° left
        danger_right = self.is_danger(head, dy, -dx)  # 90° right

        obs = [
            rel_x,
            rel_y,
            danger_front,
            danger_left,
            danger_right,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            # * Snake length is pointless in this case because it brings no importance to the agent if it is long or short because we are not giving it a reward for being long
            # // len(self.snake) / (self.grid_size * self.grid_size),
        ]
        return obs

    def _place_food(self):
        while True:
            food_position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if food_position not in self.snake:
                return food_position

    def render(self, mode: str | None = "training"):
        if mode == "training":
            return

        import pygame

        # --- Pygame graphical rendering ---
        cell_size = CELL_SIZE
        width, height = self.grid_size * cell_size, self.grid_size * cell_size

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))  # Black background

        # Draw snake
        for segment in self.snake:
            x, y = segment[1] * cell_size, segment[0] * cell_size
            pygame.draw.rect(
                self.screen, (0, 255, 0), (x, y, cell_size, cell_size), border_radius=8
            )

        # Draw food
        fx, fy = self.food[1] * cell_size, self.food[0] * cell_size
        pygame.draw.rect(
            self.screen, (255, 0, 0), (fx, fy, cell_size, cell_size), border_radius=8
        )

        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS

        # Handle quit events to avoid freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def close(self):
        pass

    # Method for playing the game (Human interaction) this is not to be used in the RL training
    def play(self):
        import time
        import keyboard

        self.reset()
        action = 0
        while True:
            self.render()
            if keyboard.is_pressed("w"):
                action = 0
            elif keyboard.is_pressed("s"):
                action = 1
            elif keyboard.is_pressed("a"):
                action = 2
            elif keyboard.is_pressed("d"):
                action = 3

            obs, reward, done, _, __ = self.step(action)
            print(f"Reward: {reward}")
            if done:
                print("Game Over!")
                break
            time.sleep(0.25)


if __name__ == "__main__":
    env = SnakeGame(grid_size=10)
    env.play()  # Start the game for human interaction
