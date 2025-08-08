import random
import torch
from collections import deque
from dqn.observation_spec import ObservationSpec

# Rewards
EAT_FOOD = +1.0
DIE = -1.0
STEP_PENALTY = -0.01
ALIVE = +0.02

# Reward scaling factors
LOW_REACHABILITY_SCALE = 0.075
ALIGN_SCALE = 0.075
DIST_SCALE = 0.1
DANGER_SCALE = 0.001

# Constants for rendering
CELL_SIZE = 20  # Size of each cell in pixels

PATTERN_UP = [
    # Y, X
    # Layer -5
    (-5, -3),
    (-5, -2),
    (-5, -1),
    (-5, 0),
    (-5, 1),
    (-5, 2),
    (-5, 3),
    # Layer -4
    (-4, -4),
    (-4, -3),
    (-4, -2),
    (-4, -1),
    (-4, 0),
    (-4, 1),
    (-4, 2),
    (-4, 3),
    (-4, 4),
    # Layer -3
    (-3, -5),
    (-3, -4),
    (-3, -3),
    (-3, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-3, 2),
    (-3, 3),
    (-3, 4),
    (-3, 5),
    # Layer -2
    (-2, -5),
    (-2, -4),
    (-2, -3),
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-2, 3),
    (-2, 4),
    (-2, 5),
    # Layer -1
    (-1, 5),
    (-1, 4),
    (-1, 3),
    (-1, 2),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (-1, -2),
    (-1, -3),
    (-1, -4),
    (-1, -5),
    # Layer 0
    (0, 5),
    (0, 4),
    (0, 3),
    (0, 2),
    (0, 1),
    (0, -1),
    (0, -2),
    (0, -3),
    (0, -4),
    (0, -5),
    # Layer 1
    (1, -5),
    (1, -4),
    (1, -3),
    (1, -2),
    (1, -1),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    # Layer 2
    (2, -5),
    (2, -4),
    (2, -3),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    # Layer 3
    (3, -4),
    (3, -3),
    (3, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    # Layer 4
    (4, -3),
    (4, -2),
    (4, -1),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
]


class SnakeGame:

    def __init__(self, grid_size=10, render_mode="training"):

        super(SnakeGame, self).__init__()
        self.grid_size = grid_size
        self.snake = [(grid_size // 2, grid_size // 2)]
        self.direction = (0, 1)  # Start moving right
        self.food = self._place_food()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.render_mode = render_mode
        self.step_count = 0
        self.reset()
        obs = self._get_observation()  # Initial observation

        # Action and Observation Spaces
        self.action_dim = 3  # Left, Straight, Right
        self.observation_space = ObservationSpec(shape=len(obs))

    def reset(self, *, seed=None, options=None):
        # Do NOT call super().reset(seed=seed)
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)
        self.food = self._place_food()
        obs = self._get_observation()

        # Or return obs, {} is for compatibility with gym API, it also allows for additional info if needed
        return obs

    def step(self, action):
        self.step_count += 1
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        idx = directions.index(self.direction)

        # Save reachable space BEFORE moving
        before_space = self.bfs_reachable_area(
            self.snake[0][0], self.snake[0][1], self.snake, self.grid_size
        )

        # Action mapping: 0 = Turn Left, 1 = Straight, 2 = Turn Right
        if action == 0:
            idx = (idx - 1) % 4
        elif action == 2:
            idx = (idx + 1) % 4
        self.direction = directions[idx]

        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )
        prev_dist = self._euclidean_dist(self.snake[0], self.food)
        new_dist = self._euclidean_dist(new_head, self.food)

        # Check death
        if new_head in self.snake or not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        ):
            return self._get_observation(), DIE, True

        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self._place_food()
            reward = EAT_FOOD
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()

            distance_delta = prev_dist - new_dist
            reward = DIST_SCALE * distance_delta

            alignment = torch.tensor(
                [
                    self.food[0] - new_head[0] / self.grid_size,
                    self.food[1] - new_head[1] / self.grid_size,
                ],
                dtype=torch.float32,
            )
            dirv = torch.tensor(
                self.direction,
                dtype=torch.float32,
            )
            reward += ALIGN_SCALE * torch.dot(alignment, dirv)

            dx, dy = self.direction
            head_x, head_y = self.snake[0]
            num_dangers = 0
            for oy, ox in PATTERN_UP:
                dy_rot, dx_rot = self.rotate_offset(oy, ox, (dx, dy))
                danger = self.is_danger((head_y, head_x), dy_rot, dx_rot)
                num_dangers += 1 if danger else 0

            reward -= DANGER_SCALE * num_dangers
            reward += STEP_PENALTY
            reward += ALIGN_SCALE * (self.direction[0] + self.direction[1])
            reward += ALIVE

            # Check reachable space AFTER move
            after_space = self.bfs_reachable_area(
                self.snake[0][0], self.snake[0][1], self.snake, self.grid_size
            )
            space_diff = after_space - before_space

            # Penalize big drops in reachable space (entering holes)
            if space_diff < -5:  # Lost more than 5 safe cells
                reward -= LOW_REACHABILITY_SCALE * abs(
                    space_diff
                )  # Scale penalty with how bad it is
        return self._get_observation(), reward, False

    def _euclidean_dist(self, a, b):
        import math

        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

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

        # Danger flags
        danger_flags = []
        for oy, ox in PATTERN_UP:
            dy_rot, dx_rot = self.rotate_offset(oy, ox, (dx, dy))
            danger = self.is_danger((head_y, head_x), dy_rot, dx_rot)
            danger_flags.append(int(danger))
        self.danger_vision_flags = danger_flags

        # RayCasts
        front, left, right = self.get_relative_directions(dx, dy)
        self.ray_directions = [front, left, right]
        self.raycasts = [
            self.raycast(head_y, head_x, *front, self.snake, self.grid_size),
            self.raycast(head_y, head_x, *left, self.snake, self.grid_size),
            self.raycast(head_y, head_x, *right, self.snake, self.grid_size),
        ]

        # NEW: Reachable space feature
        reachable_cells = self.bfs_reachable_area(
            head_y, head_x, self.snake, self.grid_size
        )
        reachable_ratio = reachable_cells / (
            self.grid_size * self.grid_size
        )  # normalize

        # Combine all observations
        obs = torch.tensor(
            [
                rel_x,
                rel_y,
                *danger_flags,
                dir_up,
                dir_down,
                dir_left,
                dir_right,
                *self.raycasts,
                reachable_ratio,
            ],
            dtype=torch.float32,
            device=self.device,
        )  # shape: [obs_dim]
        return obs

    def get_relative_directions(self, dx, dy):
        # Front is same as current direction
        front = (dx, dy)
        # Left is 90° turn counter-clockwise
        left = (-dy, dx)
        # Right is 90° turn clockwise
        right = (dy, -dx)
        return front, left, right

    def bfs_reachable_area(self, start_y, start_x, snake_body, grid_size):
        visited = set()
        q = deque()
        q.append((start_y, start_x))
        visited.add((start_y, start_x))
        obstacles = set(snake_body)

        while q:
            y, x = q.popleft()
            for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    if (ny, nx) not in visited and (ny, nx) not in obstacles:
                        visited.add((ny, nx))
                        q.append((ny, nx))
        return len(visited)  # Total reachable cells

    def raycast(self, start_y, start_x, dir_y, dir_x, snake_body, grid_size: int):
        dist = 0
        y, x = start_y, start_x

        while True:
            y += dir_y
            x += dir_x
            dist += 1

            # Wall hit
            if y < 0 or y >= grid_size or x < 0 or x >= grid_size:
                return dist

            # Snake body hit
            if (y, x) in snake_body:
                return dist

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
        if mode == "training" and self.render_mode == "training":
            return

        import pygame

        cell_size = CELL_SIZE
        width, height = self.grid_size * cell_size, self.grid_size * cell_size
        head_x, head_y = self.snake[0]
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
        spacing = 16  # Pixel distance per grid step
        for i, (dy, dx) in enumerate(PATTERN_UP):
            px = center[0] + dx * spacing
            py = center[1] + dy * spacing
            color = (255, 0, 0) if self.danger_vision_flags[i] else (80, 80, 80)
            pygame.draw.circle(self.stats_surface, color, (px, py), 8)

        # RayCasts Debugging
        if hasattr(self, "ray_directions") and hasattr(self, "raycasts"):
            for dir_idx, (dx, dy) in enumerate(self.ray_directions):
                dist = self.raycasts[dir_idx]
                for step in range(1, dist):
                    ry = head_y + dy * step
                    rx = head_x + dx * step
                    px = rx * cell_size + cell_size // 2
                    py = ry * cell_size + cell_size // 2
                    pygame.draw.circle(
                        self.screen, (150, 150, 150), (py, px), cell_size // 5
                    )

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

        self.render_mode = "human"
        self.reset()
        action = 1
        while True:
            self.render()
            if keyboard.is_pressed("a"):
                action = 0
            elif keyboard.is_pressed("d"):
                action = 2
            else:
                action = 1

            _, reward, done = self.step(action)
            print(f"Reward: {reward}")
            if done:
                print("Game Over!")
                break
            time.sleep(0.25)


if __name__ == "__main__":
    env = SnakeGame(grid_size=10)
    env.play()  # Start the game for human interaction
