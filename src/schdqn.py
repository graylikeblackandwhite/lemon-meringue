# A custom StochDQN implementation according to the paper found in [1] F. Fourati, V. Aggarwal, and M.-S. Alouini, “Stochastic Q-learning for Large Discrete Action Spaces,” May 16, 2024, arXiv: arXiv:2405.10310. doi: 10.48550/arXiv.2405.10310.
# The StochDQN is DQN but it uses a stochastic arg max
import random
import numpy as np
import torch
import network_oran
from collections import deque
from itertools import product
from torch import nn
from torch import optim



class StateTransitionTuple:
    def __init__(self, state, action, reward, next_state) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class StochDQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(StochDQNNetwork, self).__init__()
        self.device: torch.accelerator = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
        print(f"Using {self.device} device")
        self.loss: nn.MSELoss = nn.MSELoss() 
        self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.fc2: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.fcq: nn.Linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fcq(x)
        return x
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self,experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        batch = np.random.choice(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones

class StochDQNAgent:
    # Stochastic DQN Agent.
    # Flatten s(t), feed into network.
    def __init__(self, state_size: int, action_size: int, dictionary_index_to_action, simulation: network_oran.NetworkSimulation, learning_rate: float=0.001, hidden_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, use_cuda=False, deterministic=False, double=False) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.simulation = simulation
        
        self.input_size = self.state_size + len(dictionary_index_to_action[0])
        self.model: StochDQNNetwork = StochDQNNetwork(simulation.numDUs*simulation.numRUs + simulation.numDUs + simulation.numRUs, 64)
        self.target_model: StochDQNNetwork = StochDQNNetwork(simulation.numDUs*simulation.numRUs + simulation.numDUs + simulation.numRUs, 64)

        self.target_model.load_state_dict(self.model.state_dict())

        self.log2_actions = round(np.log2(action_size))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.dictionary_index_to_action = dictionary_index_to_action
        self.batch_size = 2* self.log2_actions
        self.buffer_size = 2 * self.batch_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.deterministic = deterministic
        
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print('device ', self.device)
        
        self.target_model.eval()
        
        self.double = double
        
    def stochMax(self, state_tensor, action_set, target=False):
        best_action = [action_set[0]] * state_tensor.shape[0]
        q_max = torch.tensor([-float(np.inf)] * state_tensor.shape[0], dtype=torch.float32, device=self.device)
        
        action_set = set(action_set)
        
        for action in action_set:
            action_tensor = torch.tensor(self.dictionary_index_to_action[action], dtype=torch.float32, device=self.device)
            
            action_tensor = action_tensor.repeat(state_tensor.shape[0],1)
            
            x = torch.cat((state_tensor, action_tensor), dim=1)
            
            if target:
                q_value = self.target_model(x)
            else:
                q_value = self.model(x)
            
            for i in range(state_tensor.shape[0]):
                if q_value[i] > q_max[i]:
                    q_max[i] = q_value[i].item()
                    best_action[i] = action
                    
        return q_max, best_action
        
    def selectaction(self,state,random_actions):
        if np.random.rand() <= self.epsilon or len(self.replay_buffer.buffer) < self.batch_size:
            random_index = random.randrange(self.action_size)
            return random_index
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                if len(state_tensor.shape) == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                    
                if self.deterministic:
                    _, best_action = self.stochMax(state_tensor, range(self.action_size))
                    
                else:
                    _, batch_actions, _, _, _ = self.replay_buffer.sample(self.batch_size)
                    actions_set = np.concatenate((batch_actions, random_actions), axis=0)
                    _, best_action = self.stochMax(state_tensor, actions_set)
                    
                return best_action[0]
        
    def train(self, state, action, reward, next_state, done, random_actions):
        self.replay_buffer.add((state,action,reward,next_state,done))
        
        if len(self.replay_buffer.buffer) >= self.batch_size:
            max_reps = 1
            repetition = 0
            while repetition < max_reps:
                repetition += 1
                
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(self.batch_size)
                
                state_tensor = torch.tensor(batch_states, dtype=torch.float32, device=self.device)
                next_state_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=self.device)
                batch_dones = torch.tensor([1 if done else 0 for done in batch_dones], dtype=torch.float32, device=self.device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
                batch_actions_ = torch.tensor([self.dictionary_index_to_action[action] for action in batch_actions])
                
                x = np.concatenate((state_tensor, batch_actions_), axis=1)
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
                
                actions_set = np.concatenate((batch_actions, random_actions), axis=0)
                q_max, _ = self.stochMax(next_state_tensor, actions_set, target=self.double)
                
                q_values = batch_rewards_tensor + self.gamma * q_max * (1 - batch_dones)
                
                self.optimizer.zero_grad()
                loss = self.loss_fn(q_values.unsqueeze(1), self.model(x))
                loss.background()
                self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                
    def updateTargetNetwork(self):
        self.target_model.load_state_dict(self.model.state_dict())