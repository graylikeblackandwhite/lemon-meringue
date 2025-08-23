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
    
    simulation_length = 43200
    
    
    NS_1 = NetworkSimulation(N,L,K, 1000, seed=724) # Energy efficient reward model
    NS_1.run(simulation_length)

    print(f"Control: {NS_1.get_total_energy_consumption()}")
