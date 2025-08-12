import network_oran

def main():
    NS = network_oran.networkSimulation(4,6,50,1000,0.005)
    NS.run(43200)

main()