import network_oran
import schdqn
import numpy as np
import time
from multiprocessing import Process, Lock, Manager
from waiting import wait

def randomActions(ns: network_oran.NetworkSimulation, lock):
    lock.acquire()
    try:
        print(f'NumRUs: {ns.numRUs*ns.numDUs}')
    finally:
        lock.release()
        
    while True:
        print(ns.running)
        if ns.running:
            RU_ID = np.random.randint(0, ns.numRUs*ns.numDUs-1)
            RU: network_oran.O_RU = ns.RUs[RU_ID]
            lock.acquire()
            try:
                print(f'Shutting off RU {RU_ID}')
            finally:
                lock.release()
            ns.do(network_oran.NetworkSimulationActionType.RU_SLEEP, ru=RU)

def main():
    lock = Lock()
    manager = Manager()
    namespace = manager.Namespace()
    queue = manager.Queue()
    namespace.NS = network_oran.NetworkSimulation(3,6,50,1000, queue, 0.005)
    P: Process = Process(target=randomActions, args=(namespace.NS,lock,), daemon=True, name="RandomActionProcess")
    P.start()
    namespace.NS.run(43200)
    
main()