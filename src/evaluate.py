from network_oran import *

def main():
    NS = networkSimulation(4,6,50,1000,0.005)
    NS.run(43200)

main()