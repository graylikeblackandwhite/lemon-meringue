# Test models here
# See /models for saved models

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import StochDQNAgent
from utils import NetworkSimulation

import pandas as pd

STOCH_DQN_ALPHA_1_2_BETA_0_4 = '../models/stochastic_dqn_alpha_1.2_beta_0.4.pth'

if __name__ == '__main__':
    K = 80
    L = 6
    N = 24
    
    simulation_length = 200_000
    
    agent: StochDQNAgent = StochDQNAgent(N,L,K)
    agent.load(STOCH_DQN_ALPHA_1_2_BETA_0_4)
    
    NS_1 = NetworkSimulation(N,L,K, 1000, seed=724) # Energy efficient reward model
    NS_1.alpha = 1.2
    NS_1.beta = 0.4
    NS_1.running = True
    NS_1.initialize_components()
    NS_1.simulation_length = simulation_length
    
    fronthaul_delays: list[float] = []

    for step in range(0, simulation_length):
        state = NS_1.generate_state_vector()
        fronthaul_delays.append(NS_1.calculate_average_fronthaul_delay())
        action = agent.stoch_policy(state)
        agent.interpret_action(action, NS_1)
        NS_1.step(step)
    
    print(f"Alpha 1.2 Beta 0.4: {NS_1.get_total_energy_consumption()}")
    print(f"Fronthaul Delays: {sum(fronthaul_delays)/len(fronthaul_delays)}")
    
