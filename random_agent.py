import numpy as np
from typing import List, Tuple, Any
from env import SnakeEnv


def make_env(render: bool) -> SnakeEnv:
    """Creates and returns the Snake environment"""
    mode = "human" if render else None
    return SnakeEnv(render_mode=mode)


class RandomAgent:
    """Random Policy Agent (Baseline)"""

    def __init__(self, action_space: Any):
        self.action_space = action_space

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        # Randomly sample from the action space (0: Forward, 1: Turn Right, 2: Turn Left)
        return self.action_space.sample()

    def learn(self, trajectories: List[List[Tuple[np.ndarray, int, float, np.ndarray]]]) -> None:
        # A random agent does not need a learning process
        pass


def get_rollout(agent: Any, env: SnakeEnv, greedy: bool = False) -> Tuple[
    float, List[Tuple[np.ndarray, int, float, np.ndarray]]]:
    """
    Executes a full episode and collects the trajectory for subsequent offline training or experience replay.
    Returns: (total reward, list of trajectories[(state, action, reward, next_state)])
    """
    obs, _ = env.reset()
    total_reward = 0.0
    trajectory = []

    while True:
        action = agent.act(obs, greedy=greedy)
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append((obs.copy(), action, reward, next_obs.copy()))
        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    env.close()
    return total_reward, trajectory


if __name__ == "__main__":
    env_no_render = make_env(render=False)
    agent = RandomAgent(env_no_render.action_space)

    # --- Perform 100 non-rendered game evaluations ---
    print("Running 100 non-rendered games for baseline evaluation...")
    rewards = []
    for i in range(100):
        r, _ = get_rollout(agent, env_no_render)
        rewards.append(r)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/100 | Current average reward: {np.mean(rewards):.2f}")

    print(f"\n100 game evaluation complete:")
    print(
        f"Average reward: {np.mean(rewards):.2f} | Standard deviation: {np.std(rewards):.2f} | Max score: {np.max(rewards):.0f} | Min score: {np.min(rewards):.0f}\n")

    # --- Perform 1 rendered game for visual demonstration ---
    print("Running 1 rendered game...")
    env_render = make_env(render=True)
    reward, trajectory = get_rollout(agent, env_render)
    print(f"Total reward for this rendered game: {reward:.0f} | Game steps: {len(trajectory)}")