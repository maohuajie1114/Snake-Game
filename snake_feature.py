import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Tuple, Dict, Any, Optional
from collections import deque


class SnakeEnv(gym.Env):
    """
    Reinforcement learning Snake environment, strictly compliant with the Gymnasium specification.
    Specific constraints:
    1. The board size is strictly fixed at 16 * 16.
    2. Termination conditions (Terminated) include: hitting a wall, self-collision, or winning by reaching a length of 192 (i.e., 16*16*0.75).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        # Strictly fix the board size to 16x16
        self.grid_size: int = 16
        self.window_size: int = 400
        self.cell_size: int = self.window_size // self.grid_size

        # Victory length threshold: 16 * 16 * 0.75 = 192
        self.victory_length: int = 192

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Action space: 0: Go straight, 1: Turn right, 2: Turn left
        self.action_space = spaces.Discrete(3)

        # Observation space: 14-dimensional feature vector (11 basic + 3 Flood Fill)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)

        # Internal game state variables
        self.snake: list[Tuple[int, int]] = []
        self.direction: Tuple[int, int] = (0, 0)
        self.food: Tuple[int, int] = (0, 0)
        self.score: int = 0
        self.frame_iteration: int = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize snake position (at the center) and initial length (3)
        center = self.grid_size // 2
        self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.direction = (1, 0)  # Default direction is right

        self.score = 0
        self.frame_iteration = 0
        self._place_food()

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_state(), {}

    def _place_food(self) -> None:
        """Randomly places food, ensuring it does not overlap with the snake's body"""
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.food = (x, y)
            if self.food not in self.snake:
                break

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.frame_iteration += 1

        # 1. Update movement direction based on action
        self._take_action(action)

        # 2. Move the snake's head
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        # Insert the new head, temporarily increasing snake length by 1
        self.snake.insert(0, new_head)

        reward = 0.0
        terminated = False
        truncated = False

        # 3. Death condition check: hit wall or self (note self.snake[1:] because the head was just inserted at index 0)
        is_collision = (
                new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size or
                new_head in self.snake[1:]
        )

        if is_collision:
            terminated = True
            reward = -10.0  # Trigger death penalty
            return self._get_state(), reward, terminated, truncated, {"score": self.score, "is_victory": False}

        # 4. Truncation condition: no food eaten for too long (to prevent infinite loops)
        if self.frame_iteration > 100 * len(self.snake):
            truncated = True

        # 5. Food and victory condition check
        if new_head == self.food:
            self.score += 1

            # Check if the victory condition is met (length reaches 192)
            if len(self.snake) >= self.victory_length:
                terminated = True
                reward = 50.0  # Give a large victory reward
                return self._get_state(), reward, terminated, truncated, {"score": self.score, "is_victory": True}
            else:
                reward = 10.0  # Regular reward for eating food
                self._place_food()
        else:
            self.snake.pop()  # If no food is eaten, the tail shrinks to maintain the original length
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return self._get_state(), reward, terminated, truncated, {"score": self.score, "is_victory": False}

    def _take_action(self, action: int) -> None:
        """Calculates the new absolute direction based on the relative action (0: straight, 1: right turn, 2: left turn)"""
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Down, Left, Up
        idx = clock_wise.index(self.direction)

        if action == 1:  # Turn right
            self.direction = clock_wise[(idx + 1) % 4]
        elif action == 2:  # Turn left
            self.direction = clock_wise[(idx - 1) % 4]

    def _get_free_space(self, start_pt: Tuple[int, int]) -> float:
        """
        Uses BFS (Flood Fill) to calculate the number of safely connected cells starting from start_pt.
        Returns the normalized ratio of available space (0.0 ~ 1.0).
        """
        # If the starting point is out of bounds or directly on the snake's body, the available space is 0
        # (Note: For strict accuracy, we could ignore the snake's tail self.snake[-1], as it will be vacant after the snake moves one step forward,
        # but to simplify the search and be conservatively safe, we treat the entire current snake body as an obstacle)
        if (start_pt[0] < 0 or start_pt[0] >= self.grid_size or
                start_pt[1] < 0 or start_pt[1] >= self.grid_size or
                start_pt in self.snake):
            return 0.0

        visited = set()
        queue = deque([start_pt])
        visited.add(start_pt)
        free_space = 0

        # Maximum number of cells on the entire map
        max_search = self.grid_size * self.grid_size

        while queue and free_space < max_search:
            curr = queue.popleft()
            free_space += 1

            # Iterate through the four adjacent cells (up, down, left, right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nxt = (curr[0] + dx, curr[1] + dy)
                # Check boundary conditions
                if 0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size:
                    # Check if not visited and not an obstacle
                    if nxt not in visited and nxt not in self.snake:
                        visited.add(nxt)
                        queue.append(nxt)

        # Return the normalized space ratio
        return free_space / max_search

    def _get_state(self) -> np.ndarray:
        """Extract a 14-dimensional feature vector, including basic relative features and Flood Fill spatial features"""
        head = self.snake[0]

        def is_danger(offset: Tuple[int, int]) -> bool:
            pt = (head[0] + offset[0], head[1] + offset[1])
            return (pt[0] < 0 or pt[0] >= self.grid_size or
                    pt[1] < 0 or pt[1] >= self.grid_size or
                    pt in self.snake[1:])

        dir_r, dir_d, dir_l, dir_u = (1, 0), (0, 1), (-1, 0), (0, -1)
        clock_wise = [dir_r, dir_d, dir_l, dir_u]
        idx = clock_wise.index(self.direction)

        straight = clock_wise[idx]
        right = clock_wise[(idx + 1) % 4]
        left = clock_wise[(idx - 1) % 4]

        # 1. Danger perception (1-step lookahead)
        danger_straight = int(is_danger(straight))
        danger_right = int(is_danger(right))
        danger_left = int(is_danger(left))

        # 2. Current direction
        dir_r_flag = int(self.direction == dir_r)
        dir_d_flag = int(self.direction == dir_d)
        dir_l_flag = int(self.direction == dir_l)
        dir_u_flag = int(self.direction == dir_u)

        # 3. Relative position of food
        food_left = int(self.food[0] < head[0])
        food_right = int(self.food[0] > head[0])
        food_up = int(self.food[1] < head[1])
        food_down = int(self.food[1] > head[1])

        # 4. Flood Fill connected space features
        pt_straight = (head[0] + straight[0], head[1] + straight[1])
        pt_right = (head[0] + right[0], head[1] + right[1])
        pt_left = (head[0] + left[0], head[1] + left[1])

        space_straight = self._get_free_space(pt_straight)
        space_right = self._get_free_space(pt_right)
        space_left = self._get_free_space(pt_left)

        state = [
            danger_straight, danger_right, danger_left,
            dir_r_flag, dir_d_flag, dir_l_flag, dir_u_flag,
            food_left, food_right, food_up, food_down,
            space_straight, space_right, space_left
        ]
        return np.array(state, dtype=np.float32)

    def _init_pygame(self) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.set_caption('Snake RL Env (16x16)')
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return

        self._init_pygame()
        self.window.fill((0, 0, 0))

        # Draw snake body (green)
        for pt in self.snake:
            pygame.draw.rect(self.window, (0, 255, 0),
                             pygame.Rect(pt[0] * self.cell_size, pt[1] * self.cell_size, self.cell_size,
                                         self.cell_size))

        # Draw food (red)
        pygame.draw.rect(self.window, (255, 0, 0),
                         pygame.Rect(self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size,
                                     self.cell_size))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            # Recording mode: instead of flipping the display to the real screen, convert the in-memory Surface to a NumPy matrix
            # The dimensions extracted by Pygame are (width, height, channels)
            # RecordVideo expects dimensions of (height, width, channels), so a transpose is needed
            img_array = pygame.surfarray.array3d(self.window)
            return np.transpose(img_array, axes=(1, 0, 2))
        return None

    def close(self) -> None:
        if self.window is not None:
            pygame.quit()
            self.window = None