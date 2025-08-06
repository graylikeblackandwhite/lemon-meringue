from network_oran import *

def main():
    NS = networkSimulation(2,8,50,1000,0.05)
    NS.run(500)

main()