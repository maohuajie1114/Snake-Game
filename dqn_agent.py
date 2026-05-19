import numpy as np
import random
from typing import List, Tuple, Any
from collections import deque
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -----------------------------------------
# 1. Neural Network Architecture
# -----------------------------------------
class QNetwork(nn.Module):
    """A standard fully connected neural network for the 11D state space."""

    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------------------
# 2. Replay Buffer
# -----------------------------------------
class ReplayBuffer:
    """Experience Replay Buffer Framework using deque for efficiency"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Stores one transition data"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Randomly samples a batch of data"""
        transitions = random.sample(self.buffer, batch_size)
        # Unzip the batch of transitions into individual lists
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, filepath: str) -> None:
        """Saves buffer to disk for fault tolerance."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filepath: str) -> None:
        """Loads buffer from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)


# -----------------------------------------
# 3. DQN Agent
# -----------------------------------------
class DQNAgent:
    """Deep Q-Network Agent"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr

        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995

        # Increase capacity for better DQN stability
        self.memory = ReplayBuffer(capacity=100000)

        # Hardware acceleration support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Hard copy weights to target network
        self.update_target_network()

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Selects an action using Epsilon-Greedy strategy"""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        return action

    def learn(self, trajectories: List[List[Tuple[np.ndarray, int, float, np.ndarray]]]) -> None:
        """Off-policy learning interface"""
        # 1. Push trajectory to buffer
        for trajectory in trajectories:
            for i, (s, a, r, next_s) in enumerate(trajectory):
                done = (i == len(trajectory) - 1)
                self.memory.push(s, a, r, next_s, done)

        batch_size = 1024  # Larger batch size since we learn less frequently in trajectory mode
        if len(self.memory) < batch_size:
            return

        # 2. Sample from Replay Buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Convert to PyTorch tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 3. Calculate Current Q Values
        # q_network outputs Q-values for all actions; gather picks the one for the taken action
        curr_Q = self.q_network(states_t).gather(1, actions_t)

        # 4. Calculate Target Q Values
        with torch.no_grad():
            max_next_Q = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
            # If done, (1 - dones_t) becomes 0, so target is just the reward
            target_Q = rewards_t + (1 - dones_t) * self.gamma * max_next_Q

        # 5. Compute Loss and Optimize
        loss = F.mse_loss(curr_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self) -> None:
        """Hard update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Helper to decay epsilon linearly or multiplicatively"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    # --- Fault Tolerance Methods ---
    def save_checkpoint(self, filepath: str, global_step: int, scores: list) -> None:
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'global_step': global_step,
            'scores': scores
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Tuple[int, list]:
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Successfully loaded checkpoint from {filepath}")
            return checkpoint['global_step'], checkpoint.get('scores', [])
        else:
            raise FileNotFoundError(f"No checkpoint found at {filepath}")