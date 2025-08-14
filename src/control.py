import network_oran
from schdqn import StochDQNAgent
import numpy as np

def main():
    NS: network_oran.NetworkSimulation = network_oran.NetworkSimulation(3,6,50,1000)
    agent: StochDQNAgent = StochDQNAgent(18*6+6+2*18, 18+6+18*6, simulation=NS)
    agent.train(5)
    
main()