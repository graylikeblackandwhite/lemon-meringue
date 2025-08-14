from src import network_oran
from src import schdqn
import numpy as np

def main():
    NS = network_oran.NetworkSimulation(3,6,50,1000, 0.005)
    NS.run(43200)
    
main()