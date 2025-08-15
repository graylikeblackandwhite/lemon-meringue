# A custom StochDQN implementation according to the paper found in [1] F. Fourati, V. Aggarwal, and M.-S. Alouini, “Stochastic Q-learning for Large Discrete Action Spaces,” May 16, 2024, arXiv: arXiv:2405.10310. doi: 10.48550/arXiv.2405.10310.
# The StochDQN is DQN but it uses a stochastic arg max
import random
import numpy as np
import torch
import network_oran
import time
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
        print(f"Using {self.device} device")
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
        transitions = np.random.choice(self.buffer, batch_size, replace=False)
        return transitions
                    
    def canSample(self, batch_size)->bool:
        return len(self.buffer) >= batch_size * 10

class StochDQNAgent:
    # Stochastic DQN Agent.
    # Flatten s(t), feed into network.
    def __init__(self, state_size: int, action_size: int, simulation: network_oran.NetworkSimulation, learning_rate: float=0.001, hidden_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, use_cuda=False, deterministic=False, double=False) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.simulation = simulation
        
        K = simulation.numUEs
        L = simulation.numDUs
        N = simulation.numRUs*L
        
        self.model: StochDQNNetwork = StochDQNNetwork(N*K+K*2+L+N*L, N*L+L+N)
        self.target_model: StochDQNNetwork = StochDQNNetwork(N*K+K*2+L+N*L, N*L+L+N)

        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.log2_actions = round(np.log2(action_size))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.batch_size = 2* self.log2_actions
        print(f"Batch size: {self.batch_size}")
        self.buffer_size = 2 * self.batch_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.deterministic = deterministic
        
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print('device ', self.device)
        
        self.target_model.eval()
        
        self.double = double
        
    def stochPolicy(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.simulation.numDUs*self.simulation.numRUs*self.simulation.numDUs + self.simulation.numDUs + self.simulation.numRUs, (1,1))
        else:
            # See Page 5 of the SDQN paper
            av = self.model(state).detach()
            return torch.argmax(av, dim=-1, keepdim=True)
        
    def interpretAction(self, action: torch.Tensor, simulation: network_oran.NetworkSimulation):
        int_action = action.item()
        L = simulation.numDUs
        N = simulation.numRUs*L
        
        if int_action in range(0,N-1):
            print(f"Sleep {int_action-N}th RU")
            RU: network_oran.O_RU = simulation.RUs[int_action]
            RU.sleep() if RU.status() else RU.wake()
        elif int_action in range(N, N+L):
            print(f"Sleep {int_action-N}th DU")
            DU: network_oran.O_DU = simulation.DUs[int_action-N]
            DU.sleep() if DU.status() else DU.wake()
        elif int_action in range(N+L+1, N+L+1+N*L):
            
            x = int_action-N-L
            print(f"Make connection between {x%N}th RU, {x%L}th DU")
            RU: network_oran.O_RU = simulation.RUs[x%N]
            DU: network_oran.O_DU = simulation.DUs[x%L]
            if not RU in DU.getConnectedRUs():
                RU.connectDU(DU)
        else:
            print("Do nothing")
            
    def train(self, episodes):
        returns = []
        for _ in range(episodes):
            NS: network_oran.NetworkSimulation = network_oran.NetworkSimulation(3,6,50,1000,0.0025)
            NS.running = True
            while NS.running:
                NS.initializeComponents()
                NS.simulationLength = 1000
                
                NS.updateUEs()
                state = NS.generateStateVector()
                
                ep_return = 0
                # Main loop
                for _ in range(0,NS.simulationLength):
                    action: torch.Tensor = self.stochPolicy(state)
                    #print(action.item())
                    self.interpretAction(action, NS)
                    
                    NS.mainLoopStep = _
                    time.sleep(NS.timeStepLength)
                    NS.UEConnectionTurtle.clear()
                    NS.RUDUConnectionTurtle.clear()
                    NS.SimulationStatisticsTurtle.clear()
                    NS.updateStatisticsDisplay(_)
                    NS.updateComponentConnectionDisplay()
                    
                    NS.updateUEs()
                    NS.updateTotalEnergyConsumption()
                    NS.screen.update()
                    
                    next_state: torch.Tensor = NS.generateStateVector()
                    reward: torch.Tensor = NS.calculateReward()
                    
                    exp = (state,action.reshape(1),reward.reshape(1),next_state)
                    #print(exp)

                    self.replay_buffer.add(exp)
                    
                    if self.replay_buffer.canSample(self.batch_size):
                        state_b, action_b, reward_b, next_state_b = (self.replay_buffer.sample(self.batch_size))
                        
                        print(state_b)
                        qsa_b = self.model(state_b).gather(1, action_b)
                        
                        next_qsa_b = self.target_model(next_state_b)

                        target_b = reward_b + self.gamma * next_qsa_b
                        
                        loss = nn.functional.mse_loss(qsa_b, next_qsa_b)
                    
                        self.model.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    
                    time.sleep(NS.timeStepLength)
                    NS.UEConnectionTurtle.clear()
                    NS.RUDUConnectionTurtle.clear()
                    NS.SimulationStatisticsTurtle.clear()
                    NS.updateStatisticsDisplay(_)
                    NS.updateComponentConnectionDisplay()
                    
                    NS.updateUEs()
                    NS.updateTotalEnergyConsumption()
                    state = NS.generateStateVector()
                    NS.screen.update()
                    
                    ep_return += reward.item()
                print(ep_return)
                returns.append(ep_return)
                self.epsilon = max(0, self.epsilon - self.epsilon_decay)
            if _ % 10 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
    def updateTargetNetwork(self):
        self.target_model.load_state_dict(self.model.state_dict())