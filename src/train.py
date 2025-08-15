import network_oran
from schdqn import StochDQNAgent
import numpy as np

if __name__ == '__main__':
    K = 50
    L = 6
    N = 3
    agent: StochDQNAgent = StochDQNAgent(N,L,K)
    agent.train(50000)