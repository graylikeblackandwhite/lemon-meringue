from schdqn import StochDQNAgent
import datetime
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K = 80
    L = 6
    N = 24
    agent: StochDQNAgent = StochDQNAgent(N,L,K)
    agent.train(2000)