import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys


class SnakeEnv(gym.Env):
    """
    A custom Snake reinforcement learning environment, fully compatible with the Gymnasium API.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.grid_size = 16
        self.max_steps = int(self.grid_size * self.grid_size * 0.75)  # 192
        self.window_size = 512  # Window size for Pygame rendering

        # Rendering-related variables
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Observation space: 2D numpy matrix (16, 16)
        # 0=Empty, 1=Snake body, 2=Snake head, 3=Food
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(self.grid_size, self.grid_size),
            dtype=np.uint8
        )

        # Mapping from action to coordinate change (row change, column change)
        self._action_to_direction = {
            0: (-1, 0),  # Up (row-1)
            1: (1, 0),  # Down (row+1)
            2: (0, -1),  # Left (col-1)
            3: (0, 1)    # Right (col+1)
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment state to start a new episode.
        """
        # Must call super().reset(seed=seed) to initialize the built-in random number generator self.np_random
        super().reset(seed=seed)

        self.step_count = 0

        # Initial snake position, length 3, placed in the center of the map
        # The first element of the list is the snake's head
        self.snake = [
            (self.grid_size // 2, self.grid_size // 2),
            (self.grid_size // 2, self.grid_size // 2 - 1),
            (self.grid_size // 2, self.grid_size // 2 - 2),
        ]

        # Initial direction: Right (3)
        self.current_direction = 3

        # Place food randomly
        self.food = self._place_food()

        observation = self._get_obs()
        info = self._get_info()

        # Gymnasium requires reset to return (observation, info)
        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """
        Executes one step of an action and returns the new state and reward.
        """
        self.step_count += 1

        # Handle illegal moves (prevent the snake from turning back on itself)
        # If the new action is the opposite of the current direction, ignore the new action and maintain the original direction
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if action == opposites.get(self.current_direction):
            action = self.current_direction
        else:
            self.current_direction = action

        # Calculate the new head coordinates
        head_r, head_c = self.snake[0]
        dr, dc = self._action_to_direction[action]
        new_head = (head_r + dr, head_c + dc)

        # Default settings
        reward = -0.1
        terminated = False
        truncated = False

        # Check for collision with the wall
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size):
            reward = -10.0
            terminated = True
        # Check for collision with its own body
        # (Note: If food is not eaten, the tail will move forward, so the current tail can be excluded when checking for collision)
        elif new_head in self.snake[:-1]:
            reward = -10.0
            terminated = True

        if not terminated:
            # Move the snake head to the new position
            self.snake.insert(0, new_head)

            # Check if food is eaten
            if new_head == self.food:
                reward = 10.0
                # If food is eaten, do not remove the tail and place new food
                self.food = self._place_food()
            else:
                # If food is not eaten, remove the tail to maintain the same length
                self.snake.pop()

        # Check if the truncation condition is met (Truncated)
        if self.step_count >= self.max_steps:
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        # Gymnasium requires step to return (obs, reward, terminated, truncated, info)
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Generates and returns the current observation space (2D numpy matrix)"""
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Draw snake body (1)
        for r, c in self.snake:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                obs[r, c] = 1

        # Draw snake head (2) - overwrites the first body segment
        head_r, head_c = self.snake[0]
        if 0 <= head_r < self.grid_size and 0 <= head_c < self.grid_size:
            obs[head_r, head_c] = 2

        # Draw food (3)
        obs[self.food[0], self.food[1]] = 3

        return obs

    def _get_info(self):
        """Returns optional debugging information"""
        return {"snake_length": len(self.snake)}

    def _place_food(self):
        """Randomly finds an empty spot not occupied by the snake to place food"""
        while True:
            # Use Gymnasium's built-in self.np_random to ensure controllable randomness
            r = self.np_random.integers(0, self.grid_size)
            c = self.np_random.integers(0, self.grid_size)
            if (r, c) not in self.snake:
                return (r, c)

    def render(self):
        """Renders the environment"""
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("RL Snake Environment")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((30, 30, 30))  # Dark background

        cell_size = self.window_size / self.grid_size

        # Draw food (red)
        food_rect = pygame.Rect(
            self.food[1] * cell_size, self.food[0] * cell_size, cell_size, cell_size
        )
        pygame.draw.rect(canvas, (255, 50, 50), food_rect)

        # Draw snake
        for i, (r, c) in enumerate(self.snake):
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            if i == 0:
                pygame.draw.rect(canvas, (50, 255, 50), rect)  # Snake head (bright green)
            else:
                pygame.draw.rect(canvas, (50, 200, 50), rect)  # Snake body (dark green)
            # Add grid line effect
            pygame.draw.rect(canvas, (20, 20, 20), rect, 1)

        if self.render_mode == "human":
            # Must handle the pygame event queue, otherwise the window will become unresponsive
            pygame.event.pump()
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            # Control the frame rate
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            # Return a numpy matrix for video recording, requires transposing axes to match the standard RGB format (H, W, C)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Closes the environment and cleans up Pygame resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# ==========================================
# Test code: Verify environment interface and rendering
# ==========================================
if __name__ == "__main__":
    print("Initializing SnakeEnv environment...")
    # For training only, set render_mode=None to improve performance
    env = SnakeEnv(render_mode="human")

    # You can also use rgb_array to test data output:
    # env = SnakeEnv(render_mode="rgb_array")

    # Verify if the environment is fully compatible with the Gym API
    from gymnasium.utils.env_checker import check_env

    check_env(env)
    print("Environment compatibility check passed!")

    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}, Initial info: {info}")

    for step in range(100):
        # Sample a random action (0:Up, 1:Down, 2:Left, 3:Right)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step: {step + 1} | Action: {action} | Reward: {reward:.1f} | "
              f"Terminated: {terminated} | Truncated: {truncated} | Length: {info['snake_length']}")

        if terminated or truncated:
            print("Game over, resetting the environment.")
            obs, info = env.reset()
            # Just for testing continuity, you can continue the loop or break out directly
            break

    env.close()
    print("Test finished.")