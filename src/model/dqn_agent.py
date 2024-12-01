import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.model.qNetwork import QNetwork

class DQNAgent:
    """Deep Q Network Agent that selections actions based on generated Q values."""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        #Hyper Params
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.replay_buffer = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 128
        
        self.reward_scale = 1e-4  # Scale factor for rewards
        
        #state normalization
        self.state_mean = None
        self.state_std = None
        
        # tracking
        self.q_values_history = []
        self.losses_history = []
        self.epsilons_history = []
        self.rewards_history = []
        
        self.grad_clip = 1.0

    def normalize_state(self, state):
        state = np.array(state)
        if self.state_mean is None:
            self.state_mean = np.mean(state)
            self.state_std = np.std(state) + 1e-8
        return (state - self.state_mean) / self.state_std

    def scale_reward(self, reward):
        return reward * self.reward_scale

    def select_action(self, state, train=True):
        state = self.normalize_state(state)
        if train and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        reward = self.scale_reward(reward)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states), torch.LongTensor(actions),
                torch.FloatTensor(rewards), torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_experiences()
        
        #Double DQN 
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1]
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        #Huber loss
        loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values.detach())
        
        # tracking
        self.q_values_history.append(current_q_values.mean().item())
        self.losses_history.append(loss.item())
        self.epsilons_history.append(self.epsilon)

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        
        self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        tau = 0.001
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()