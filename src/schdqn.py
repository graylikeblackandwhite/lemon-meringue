# A custom StochDQN implementation according to the paper found in [1] F. Fourati, V. Aggarwal, and M.-S. Alouini, “Stochastic Q-learning for Large Discrete Action Spaces,” May 16, 2024, arXiv: arXiv:2405.10310. doi: 10.48550/arXiv.2405.10310.
# The StochDQN is DQN but it uses a stochastic arg max
from datetime import datetime
import copy
import random
import numpy as np
import torch
import network_oran
import time
import pandas as pd
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
    def __init__(self, input_size: int, output_size: int) -> None:
        super(StochDQNNetwork, self).__init__()
        self.device: torch.accelerator = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
        hidden_size = int((2/3) * input_size) + output_size
        self.loss: nn.MSELoss = nn.MSELoss() 
        self.fc1: nn.Linear = nn.Linear(input_size,  hidden_size)
        self.fc2: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.fcq: nn.Linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fcq(x)
        return x
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        torch.set_printoptions(threshold=10000)
        
    def add(self,experience):
        self.buffer.append(experience)
        
    def __sizeof__(self) -> int:
        return len(self.buffer)
        
    def sample(self, batch_size):
        assert self.canSample(batch_size)
        transitions = random.sample(self.buffer, batch_size)
        state_b, action_b, reward_b, next_state_b = zip(*transitions)
    
        state_b = torch.stack(state_b)                           # (batch, state_dim)
        action_b = torch.stack(action_b).long().unsqueeze(1)     # (batch, 1)
        reward_b = torch.stack(reward_b).float().unsqueeze(1)    # (batch, 1)
        next_state_b = torch.stack(next_state_b)                 # (batch, state_dim)
        
        return state_b, action_b, reward_b, next_state_b
                    
    def canSample(self, batch_size)->bool:
        return len(self.buffer) >= batch_size * 10

class StochDQNAgent:
    # Stochastic DQN Agent.
    # Flatten s(t), feed into network.
    def __init__(self, numRUs:int, numDUs: int, numUEs: int, learning_rate: float=1e-4, gamma=0.99, epsilon=0.9, epsilon_decay=0.005, epsilon_min=0.05, use_cuda=False) -> None:
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print('device ', self.device)
        
        self.K = numUEs
        self.L = numDUs
        self.N = numRUs*self.L
        
        self.model: StochDQNNetwork = StochDQNNetwork(self.N*self.K+self.K*2+self.L+self.N*self.L, self.N*self.L+self.L+self.N)
        self.target_model: StochDQNNetwork = copy.deepcopy(self.model).to(self.device).eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.log2_actions = round(np.log2(self.N+self.L+self.N*self.L))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.batch_size = 8* self.log2_actions
        print(f"Batch size: {self.batch_size}")
        self.replay_buffer = ReplayBuffer(25000000)
        
    def stochPolicy(self, state):
        rand = torch.rand(1)
        if rand.item() > 1-self.epsilon:
            A = self.L*self.N + self.L + self.N
            exploited_action = torch.randint(0, A, (1,1))
            return exploited_action
        else:
            # See Page 5 of the SDQN paper
            A = self.L*self.N + self.L + self.N
            av = self.model(state).detach()
            if self.replay_buffer.canSample(self.batch_size):
                R = torch.randint(0,A,(1,self.batch_size))
                _, action_sample, _, _ = self.replay_buffer.sample(self.batch_size)
                action_sample = action_sample.reshape(1,self.batch_size)
                C = torch.flatten(torch.cat((R,action_sample),0))
                maxq = 0
                for q in C:
                    if av[q]>av[maxq]:
                        maxq = q
                maxq = torch.tensor(maxq)
                return maxq
            else:
                return torch.argmax(av, dim=-1, keepdim=True)
    def interpretAction(self, action: torch.Tensor, simulation: network_oran.NetworkSimulation):
        int_action = action.item()
        L = simulation.numDUs
        N = simulation.numRUs*L
        
        if int_action in range(0,N-1):
            #print(f"Sleep {int_action}th RU")
            RU: network_oran.O_RU = simulation.RUs[int_action]
            RU.sleep() if RU.status() else RU.wake()
        elif int_action in range(N, N+L):
            #print(f"Sleep {int_action-N}th DU")
            DU: network_oran.O_DU = simulation.DUs[int_action-N]
            DU.sleep() if DU.status() and len(DU.getConnectedRUs()) == 0 else DU.wake()
        elif int_action in range(N+L+1, N+L+1+N*L):
            x = int_action-N-L
            #print(f"Make connection between {x%N}th RU, {x%L}th DU")
            RU: network_oran.O_RU = simulation.RUs[x%N]
            DU: network_oran.O_DU = simulation.DUs[x%L]
            if not RU in DU.getConnectedRUs():
                RU.connectDU(DU)
        else:
            pass
        
    def writeEpisodeResults(self, episodes, simulationLength, rewards)->None:
        data = {
            'Episode': [i+1 for i in range(episodes)],
            'Simulation Length (seconds)': [simulationLength for _ in range(episodes)],
            'Reward' : rewards
            }
        df = pd.DataFrame(data)
        df.to_csv(f'../data/model_output{datetime.now()}')
            
    def train(self, episodes):
        returns = []
        for episode in range(episodes):
            NS: network_oran.NetworkSimulation = network_oran.NetworkSimulation(3,6,50,1000,0)
            NS.running = True
            while NS.running:
                NS.initializeComponents()
                NS.simulationLength = 500
                
                NS.step(0)
                state = NS.generateStateVector()
                
                ep_return = 0
                # Main loop
                for step in range(1,NS.simulationLength):
                    action: torch.Tensor = self.stochPolicy(state)
                    self.interpretAction(action, NS)
                    
                    time.sleep(NS.timeStepLength)
                    NS.step(step)
                    
                    next_state: torch.Tensor = NS.generateStateVector()
                    reward: torch.Tensor = NS.calculateReward()
                    
                    action = action.view(-1)
                    reward = torch.tensor([reward])
                    
                    exp = [state,action,reward,next_state]
                    self.replay_buffer.add(exp)
                    
                    if self.replay_buffer.canSample(self.batch_size):
                        state_b, action_b, reward_b, next_state_b = self.replay_buffer.sample(self.batch_size)
                        action_b = action_b.reshape(self.batch_size,1)
                        qsa_b = self.model(state_b).gather(1, action_b)
                        
                        next_qsa_b = self.target_model(next_state_b)
                        next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                        
                        target_b = reward_b + self.gamma * next_qsa_b
                        
                        self.loss = nn.functional.mse_loss(qsa_b, target_b)
                    
                        self.model.zero_grad()
                        self.loss.backward()
                        self.optimizer.step()
                    
                    NS.step(step)
                    if step % 50 == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                    state = NS.generateStateVector()
                    ep_return += reward.item()
                print(ep_return)
                NS.running = False
                returns.append(ep_return)
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
                break
            for turtle in NS.screen.turtles():
                del turtle
            NS.screen.clearscreen()
        self.writeEpisodeResults(episodes,1000,returns)
        self.model.eval()
        torch.save(self.model.state_dict(), f"../models/stochdqn_{datetime.now()}.pth")
