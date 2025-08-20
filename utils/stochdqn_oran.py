# A custom StochDQN implementation according to the paper found in [1] F. Fourati, V. Aggarwal, and M.-S. Alouini, “Stochastic Q-learning for Large Discrete Action Spaces,” May 16, 2024, arXiv: arXiv:2405.10310. doi: 10.48550/arXiv.2405.10310.
# The StochDQN is DQN but it uses a stochastic arg max as well as a stochastic optimizer
import copy
import random
import numpy as np
import torch
import utils.network_oran as network_oran
import pandas as pd
from collections import deque
from torch import nn
from torch import optim
from datetime import datetime

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
        assert self.can_sample(batch_size)
        transitions = random.sample(self.buffer, batch_size)
        state_b, action_b, reward_b, next_state_b = zip(*transitions)
    
        state_b = torch.stack(state_b)                           # (batch, state_dim)
        action_b = torch.stack(action_b).long().unsqueeze(1)     # (batch, 1)
        reward_b = torch.stack(reward_b).float().unsqueeze(1)    # (batch, 1)
        next_state_b = torch.stack(next_state_b)                 # (batch, state_dim)
        
        return state_b, action_b, reward_b, next_state_b
                    
    def can_sample(self, batch_size)->bool:
        return len(self.buffer) >= batch_size * 10

class StochDQNAgent:
    # Stochastic DQN Agent.
    # Flatten s(t), feed into network.
    def __init__(self, numRUs:int, numDUs: int, numUEs: int, learning_rate: float=1e-4, gamma=0.99, epsilon=0.9, epsilon_decay=1e-3, epsilon_min=0.05, epsilon_max =1.0, use_cuda=False, simulation_length_in_training: int=100) -> None:
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print('device ', self.device)
        
        self.K = numUEs
        self.L = numDUs
        self.N = numRUs
        self.simulation_length_in_training = simulation_length_in_training
        self.model: StochDQNNetwork = StochDQNNetwork(self.N*self.K+self.K*2+self.L+self.N*self.L, self.N*self.L+self.L+self.N+1)
        self.target_model: StochDQNNetwork = copy.deepcopy(self.model).to(self.device).eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.log2_actions = round(np.log2(self.N+self.L+self.N*self.L))
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.batch_size = 8* self.log2_actions
        print(f"Batch size: {self.batch_size}")
        self.replay_buffer = ReplayBuffer(20000)
        
    def stoch_policy(self, state: torch.Tensor) -> torch.Tensor:
        rand = torch.rand(1)
        if rand.item() > 1-self.epsilon:
            A = self.L*self.N + self.L + self.N
            exploited_action = torch.randint(0, A, (1,1))
            return exploited_action
        else:
            # See Page 5 of the SDQN paper
            A = self.L*self.N + self.L + self.N
            av = self.model(state).detach()
            if self.replay_buffer.can_sample(self.batch_size):
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
            
    def interpret_action(self, action: torch.Tensor, simulation: network_oran.NetworkSimulation):
        int_action = action.item()
        L = simulation.num_dus
        N = simulation.num_rus

        if int_action in range(0,N-1):
            #print(f"Sleep {int_action}th RU")
            ru: network_oran.O_RU = simulation.rus[int_action]
            ru.sleep() if ru.status() else ru.wake()
        elif int_action in range(N, N+L):
            #print(f"Sleep {int_action-N}th DU")
            du: network_oran.O_DU = simulation.dus[int_action-N]
            du.sleep() if du.status() and len(du.get_connected_rus()) == 0 else du.wake()
        elif int_action in range(N+L+1, N+L+1+N*L):
            x = int_action-N-L
            #print(f"Make connection between {x%N}th RU, {x%L}th DU")
            ru: network_oran.O_RU = simulation.rus[x%N]
            du: network_oran.O_DU = simulation.dus[x%L]
            if not ru in du.get_connected_rus():
                ru.connect_du(du)
        else:
            pass
        
    def writeEpisodeResults(self, episodes: int, simulation_length: int, rewards: list[float])->None:
        data = {
            'Episode': [i+1 for i in range(episodes)],
            'Simulation Length (seconds)': [simulation_length for _ in range(episodes)],
            'Reward' : rewards
            }
        df = pd.DataFrame(data)
        df.to_csv(f'../data/model_output{datetime.now()}')
            
    def train(self, episodes):
        returns = []
        losses = []
        for episode in range(1,episodes+1):
            print(f"Episode {episode}/{episodes}")
            NS: network_oran.NetworkSimulation = network_oran.NetworkSimulation(self.N, self.L, self.K,1000,0)
            NS.running = True
            while NS.running:
                NS.initialize_components()
                NS.simulation_length = self.simulation_length_in_training

                NS.step(0)
                state = NS.generate_state_vector()

                ep_return = 0
                # Main loop
                for step in range(1,NS.simulation_length):
                    action: torch.Tensor = self.stoch_policy(state)
                    self.interpret_action(action, NS)
                    reward: torch.Tensor = NS.calculate_reward()
                    NS.step(step)

                    next_state: torch.Tensor = NS.generate_state_vector()

                    action = action.view(-1)
                    reward = torch.tensor([reward])
                    
                    exp = [state,action,reward,next_state]
                    self.replay_buffer.add(exp)
                    
                    if self.replay_buffer.can_sample(self.batch_size):
                        self.model.zero_grad()
                        state_b, action_b, reward_b, next_state_b = self.replay_buffer.sample(self.batch_size)
                        action_b = action_b.reshape(self.batch_size,1)
                        qsa_b = self.model(state_b).gather(1, action_b)
                        
                        next_qsa_b = self.target_model(next_state_b)
                        next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

                        if step == NS.simulation_length-1:
                            target_b = reward_b
                        else:
                            target_b = reward_b + self.gamma * next_qsa_b
                        
                        loss = nn.functional.mse_loss(qsa_b, target_b)
                        loss.backward()
                        self.optimizer.step()
                        
                    NS.step(step)
                    state = NS.generate_state_vector()
                    if (episode*self.simulation_length_in_training + step) % 200 == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                    
                    ep_return += reward.item()
                print(f"Episode {episode}/{episodes} - Return: {ep_return}")
                NS.running = False
                returns.append(ep_return)
                
                self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)
                break
            for turtle in NS.screen.turtles():
                del turtle
            NS.screen.clearscreen()
            if episode % 50 == 0:
                self.writeEpisodeResults(episode,self.simulation_length_in_training,returns)
        
        self.model.eval()
        torch.save(self.model.state_dict(), f"../models/stochdqn_{datetime.now()}.pth")
        return returns
