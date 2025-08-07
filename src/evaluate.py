from network_oran import *

def main():
    NS = networkSimulation(2,6,20,1000,0.025)
    NS.run(5000)

main()