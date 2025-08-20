import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import StochDQNAgent

if __name__ == '__main__':
    K = 80
    L = 6
    N = 24
    agent: StochDQNAgent = StochDQNAgent(N,L,K)
    agent.train(8000)