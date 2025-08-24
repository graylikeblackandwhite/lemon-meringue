# Test models here
# See /models for saved models

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import StochDQNAgent
from utils import NetworkSimulation

import pandas as pd

STOCH_DQN_ALPHA_1_2_BETA_0_4 = '../models/stochastic_dqn_alpha_1.2_beta_0.4.pth'
STOCH_DQN_ALPHA_0_6_BETA_0_4 = '../models/stochastic_dqn_alpha_0.6_beta_0.4.pth'

if __name__ == '__main__':
    K = 80
    L = 6
    N = 24
    
    simulation_length = 200_000
    
    agent: StochDQNAgent = StochDQNAgent(N,L,K)
    agent.load(STOCH_DQN_ALPHA_0_6_BETA_0_4)
    
    NS_2 = NetworkSimulation(N,L,K, 1000, seed=724) # Balanced reward model
    NS_2.alpha = 0.6
    NS_2.beta = 0.4
    NS_2.running = True
    NS_2.initialize_components()
    NS_2.simulation_length = simulation_length
    
    fronthaul_delays: list[float] = []

    for step in range(0, simulation_length):
        state = NS_2.generate_state_vector()
        fronthaul_delays.append(NS_2.calculate_average_fronthaul_delay())
        action = agent.stoch_policy(state)
        agent.interpret_action(action, NS_2)
        NS_2.step(step)

    print(f"Alpha 0.6 Beta 0.4: {NS_2.get_total_energy_consumption()}")
    print(f"Fronthaul Delays: {sum(fronthaul_delays)/len(fronthaul_delays)}")
    
    
