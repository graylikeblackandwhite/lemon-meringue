# STOCHASTIC DEEP Q-LEARNING IMPLEMENTATION
# I used PyTorch and numpy here (go figure)
# Ideas for model implementation:
# Instead of having the neural network output an entire flattened vector (\mathcal{N} \times L + \mathcal{N} + L), maybe we can try just outputting a vector with four members \mathcal{X} = [d_i, r_j, u, v] where we have the pair <d_i, r_j> telling us to reassign d_i to r_j, OR we have u = the id of some RU in the simulation, and if we see this, that means "change the sleep state of this RU", same thing for v, just for the DU. this way the output of the neural network will be far smaller.

import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
from collections import namedtuple, deque

seed = 42
random.seed(seed)

# Hyperparameters
batch_size = 128
gamma = 0.99
alpha = 3e-4
epsilon_start = 0.9
epsilon_end = 0.01
epsilon_decay = 2500
tau = 0.005

# Stoch DQN

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory,batch_size) # type: ignore
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)