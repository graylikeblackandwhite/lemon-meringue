# A custom StochDQN implementation according to the paper found in [1] F. Fourati, V. Aggarwal, and M.-S. Alouini, “Stochastic Q-learning for Large Discrete Action Spaces,” May 16, 2024, arXiv: arXiv:2405.10310. doi: 10.48550/arXiv.2405.10310.
# The StochDQN is DQN but it uses a stochastic arg max

import numpy as np
import torch
import network_oran
from collections import deque, namedtuple
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
print(f"Using {device} device")

class StateTransitionTuple:
    def __init__(self, state, action, reward, next_state) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class StochDQNNetwork:
    def __init__(self, state_size, action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        
        self.network = nn.Sequential(
            nn.Linear(state_size, state_size),
            nn.ReLU(),
            nn.Linear(state_size,state_size),
            nn.ReLU(),
            nn.Linear(state_size,action_size)
        )

class StochDQNAgent:
    # Stochastic DQN Agent.
    # Flatten s(t), feed into network.
    def __init__(self, state_size, action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        
        self.learning_rate = 0.001
        self.replay_buffer = deque(maxlen=2000)
        self.update_rate = 0.2
        
        self.stochDQNNetwork = StochDQNNetwork(state_size, action_size)
        
    def stochArgMax(self):
        R = np.random.choice(self.replay_buffer, np.ceil(np.log(self.action_size)))
        C = np.random.sample(np.ceil(np.log(self.action_size)))
        
    def chooseRandomAction(self):
        pass
        
    def stochasticPolicy(self, epsilon_s):
        #TODO: make more precise
        if np.random.uniform(0,1) <= epsilon_s:
            self.chooseRandomAction()
        else:
            # stoch arg max a in A Q(s,a)
            self.stochArgMax()
        
    def train(self, episodes):
        for _ in range(episodes):
            NS = network_oran.networkSimulation(int(np.random.uniform(3,6)),int(np.random.uniform(3,6)),int(np.random.uniform(12,50)),1000,0.05)
            s = NS.generateStateVector()
            while NS.mainLoopStep < 1000:
                chosen_action = self.stochasticPolicy(0.05)
                R = NS.calculateReward(s,chosen_action)
                memory = StateTransitionTuple(s,chosen_action,R,NS.generateStateVector())
                self.replay_buffer.append(memory)
    