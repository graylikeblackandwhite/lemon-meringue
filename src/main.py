from numpy import *

# CONSTANTS

c = 299792458 # speed of light
P_0 = 30.0 # in dBm


class O_DU:
    def __init__(self, px, py):
        self.x = px
        self.y = py
        

class O_RU:
    def __init__(self, px, py):
        self.x = px
        self.y = py

class networkSimulation:
    def __init__(self, n, m, k, s, dt=0.1):
        self.numRUs = n
        self.numDUs = m
        self.numUEs = k
        self.simulationSideLength = s # in meters
        self.timeStepLength = dt # amount of time one frame goes for
        pass

def fspl(d, f):
    return 20*log10(d)+20*log10(f)+20*log10(4*pi/c)

def rss(d,f):
    # Returns Received Signal Strength given a distance in meters and a frequency in Gigahertz.
    return P_0-20*log10((4*pi*d)/(c/f))

