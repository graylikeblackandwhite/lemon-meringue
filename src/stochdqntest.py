import network_oran
from schdqn import StochDQNAgent
import numpy as np

def main():
    K = 50
    L = 6
    N = 3
    agent: StochDQNAgent = StochDQNAgent(N,L,K)
    agent.train(50000)
    
main()