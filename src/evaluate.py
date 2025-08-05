from network_oran import *

def main():
    NS = networkSimulation(6,3,12,1000)
    NS.run(500)

main()