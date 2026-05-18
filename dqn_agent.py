import numpy as np
from typing import List, Tuple, Any


class ReplayBuffer:
    """Experience Replay Buffer Framework"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Stores one transition data (Transition)"""
        # TODO: Implement a circular queue or deque to store experiences
        pass

    def sample(self, batch_size: int) -> Tuple:
        """Randomly samples a batch of data"""
        # TODO: Randomly draw batch_size data from the buffer
        raise NotImplementedError("Experience sampling logic not yet implemented")

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent Skeleton"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr

        # Exploration and exploitation (epsilon-greedy) related parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Initialize the experience replay buffer
        self.memory = ReplayBuffer(capacity=10000)

        # TODO: Initialize your PyTorch / TensorFlow neural network here
        # self.q_network = YourNeuralNet(state_dim, action_dim)
        # self.target_network = YourNeuralNet(state_dim, action_dim)
        # self.optimizer = ...

        # TODO: Initialize the weights of the target_network to be the same as the q_network
        pass

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Selects an action based on the current state.
        If greedy=False, uses the Epsilon-Greedy strategy for exploration.
        If greedy=True, purely exploits by choosing the action with the maximum Q-value from the current Q-network.
        """
        # TODO:
        # 1. Generate a random number between (0,1)
        # 2. If the random number < self.epsilon and not greedy: return a random action (exploration)
        # 3. Otherwise: pass obs to the Q-network, use argmax to extract the action with the largest Q-value and return it
        raise NotImplementedError("Action selection logic not yet implemented")

    def learn(self, trajectories: List[List[Tuple[np.ndarray, int, float, np.ndarray]]]) -> None:
        """
        Off-policy learning interface.
        You can choose to push the entire trajectory into the buffer, and then perform one or more gradient descent updates.
        """
        # 1. Put the trajectory data into the replay buffer
        for trajectory in trajectories:
            for i, (s, a, r, next_s) in enumerate(trajectory):
                # Determine if the current step is the last step
                done = (i == len(trajectory) - 1)
                self.memory.push(s, a, r, next_s, done)

        # 2. Check if the amount of data is sufficient for an update
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        # 3. Sample from the replay buffer
        # batch = self.memory.sample(batch_size)

        # TODO: Core Q-learning update logic
        # 1. Calculate Target Q: Target = r + gamma * max(target_net(next_s)) (note that only r is calculated when done)
        # 2. Calculate Current Q: Current = q_net(s)[a]
        # 3. Calculate Loss (e.g., MSELoss or SmoothL1Loss)
        # 4. Zero the optimizer gradients, backpropagate with loss.backward(), and update parameters with optimizer.step()
        raise NotImplementedError("DQN gradient descent update logic not yet implemented")

    def update_target_network(self) -> None:
        """Hard update (or soft update) the weights of the Q-network to the target network"""
        # TODO: e.g., self.target_network.load_state_dict(self.q_network.state_dict())
        pass