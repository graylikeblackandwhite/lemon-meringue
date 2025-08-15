from schdqn import StochDQNAgent
import datetime
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K = 80
    L = 6
    N = 6
    agent: StochDQNAgent = StochDQNAgent(N,L,K)
    returns = agent.train(1000)
    
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Returns Over Episodes')
    plt.show()
    plt.savefig(f'training_returns_{datetime.datetime.now()}.png')