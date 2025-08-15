from schdqn import StochDQNAgent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K = 50
    L = 6
    N = 3
    agent: StochDQNAgent = StochDQNAgent(N, L, K)
    
    # Train the agent for a specified number of episodes
    num_episodes = 10
    returns = agent.train(num_episodes)
    
    # Plot the training results
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Returns Over Episodes')
    plt.show()