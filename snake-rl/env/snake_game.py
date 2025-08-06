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
DANGER_SCALE = 0.01

# Constants for rendering
CELL_SIZE = 20  # Size of each cell in pixels

PATTERN_UP = [
    # Y, X
    # Up
    (-3, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-3, 2),
    (-2, 1),
    (-2, 0),
    (-2, -1),
    (-1, 0),
    # Left
    (2, -3),
    (1, -3),
    (0, -3),
    (-1, -3),
    (-2, -3),
    (1, -2),
    (0, -2),
    (-1, -2),
    (0, -1),
    # Right
    (-2, 3),
    (-1, 3),
    (0, 3),
    (1, 3),
    (2, 3),
    (-1, 2),
    (0, 2),
    (1, 2),
    (0, 1),
]


class SnakeGame(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode="training"):
        super(SnakeGame, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(3)  # Left, Straight, Right
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

        # Directions in clockwise order: UP, RIGHT, DOWN, LEFT
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # Find current direction index
        idx = directions.index(self.direction)

        # Action mapping: 0 = Turn Left, 1 = Straight, 2 = Turn Right
        if action == 0:  # Turn Left
            idx = (idx - 1) % 4
        elif action == 2:  # Turn Right
            idx = (idx + 1) % 4
        # action == 1 → Straight, idx stays the same

        # Update direction
        self.direction = directions[idx]

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

            dx, dy = self.direction
            head_x, head_y = self.snake[0]

            # Calculate dangers based on the observation space
            num_dangers = 0
            for oy, ox in PATTERN_UP:
                dy_rot, dx_rot = self.rotate_offset(oy, ox, (dx, dy))
                danger = self.is_danger((head_y, head_x), dy_rot, dx_rot)
                num_dangers += 1 if danger else 0

            reward -= DANGER_SCALE * num_dangers  # Penalty for danger
            reward += STEP_PENALTY  # Small penalty for each step taken
            reward += ALIGN_SCALE * (self.direction[0] + self.direction[1])
            reward += ALIVE  # Small positive reward for moving

        # c_step += 1
        return self._get_observation(), reward, False, False, {}

    def _euclidean_dist(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def is_danger(self, from_pos, offset_y, offset_x):
        y, x = from_pos
        next_y = y + offset_y
        next_x = x + offset_x
        return (
            not (0 <= next_y < self.grid_size and 0 <= next_x < self.grid_size)
            or (next_y, next_x) in self.snake
        )

    def _get_observation(self):
        head_y, head_x = self.snake[0]
        food_y, food_x = self.food

        # Relative position to food
        rel_x = (food_x - head_x) / self.grid_size
        rel_y = (food_y - head_y) / self.grid_size

        dx, dy = self.direction

        # One-hot direction
        dir_up = int((dx, dy) == (-1, 0))
        dir_down = int((dx, dy) == (1, 0))
        dir_left = int((dx, dy) == (0, -1))
        dir_right = int((dx, dy) == (0, 1))

        # === Custom Vision Pattern (relative to facing UP) ===

        # Compute danger flags
        danger_flags = []
        for oy, ox in PATTERN_UP:
            dy_rot, dx_rot = self.rotate_offset(oy, ox, (dx, dy))
            danger = self.is_danger((head_y, head_x), dy_rot, dx_rot)
            danger_flags.append(int(danger))

        self.danger_vision_flags = danger_flags

        # Combine all observations
        obs = [
            rel_x,
            rel_y,
            *danger_flags,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
        ]
        return np.array(obs, dtype=np.float32)

    # Rotate offset based on current facing direction
    def rotate_offset(self, off_y, off_x, facing):
        if facing == (-1, 0):
            return off_y, off_x  # UP
        elif facing == (1, 0):
            return -off_y, -off_x  # DOWN
        elif facing == (0, 1):
            return off_x, -off_y  # RIGHT
        elif facing == (0, -1):
            return -off_x, off_y  # LEFT
        else:
            raise ValueError(f"Invalid direction: {facing}")

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

        cell_size = CELL_SIZE
        width, height = self.grid_size * cell_size, self.grid_size * cell_size
        stats_size = 200  # Width of stats panel

        # --- Main game window ---
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((width + stats_size, height))
            pygame.display.set_caption("Snake RL + Stats")
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))

        # --- Draw snake ---
        for segment in self.snake:
            x, y = segment[1] * cell_size, segment[0] * cell_size
            pygame.draw.rect(
                self.screen, (0, 255, 0), (x, y, cell_size, cell_size), border_radius=8
            )

        # --- Draw food ---
        fx, fy = self.food[1] * cell_size, self.food[0] * cell_size
        pygame.draw.rect(
            self.screen, (255, 0, 0), (fx, fy, cell_size, cell_size), border_radius=8
        )

        # --- Stats panel ---
        self.stats_surface = pygame.Surface((stats_size, height))
        self.stats_surface.fill((30, 30, 30))  # Background
        center = (stats_size // 2, height // 2)
        pygame.draw.circle(self.stats_surface, (0, 255, 0), center, 6)  # Snake head

        # Matching the vision pattern (same as PATTERN_UP used in get_observation)

        # Render each tile from vision_flags around the center
        spacing = 20  # Pixel distance per grid step
        for i, (dy, dx) in enumerate(PATTERN_UP):
            px = center[0] + dx * spacing
            py = center[1] + dy * spacing
            color = (255, 0, 0) if self.danger_vision_flags[i] else (80, 80, 80)
            pygame.draw.circle(self.stats_surface, color, (px, py), 10)

        # Combine game + stats
        combined = pygame.Surface((width + stats_size, height))
        combined.blit(self.screen, (0, 0))
        combined.blit(self.stats_surface, (width, 0))
        pygame.display.get_surface().blit(combined, (0, 0))
        pygame.display.flip()
        self.clock.tick(10)

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
