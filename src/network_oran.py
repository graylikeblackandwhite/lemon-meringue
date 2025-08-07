#   Simulation software implementation
#   Written by Gray
#   TODO: Clean up code
#   In this sim, we're using FR2, 28GHz

from numpy import * # pyright: ignore[reportWildcardImportFromLibrary]
from turtle import * # pyright: ignore[reportWildcardImportFromLibrary]
from enum import Enum
import time
import typing

seed: float = 427
rng: random.Generator = random.default_rng(seed=seed)

# CONSTANTS

c: float = 299792458 # speed of light
P_0: float = 30.0 # in dBm
E_S_DU: float = 2 # megajoules, how much energy the DU consumes by staying active
E_S_RU: float = 1 # megajoules, how much energy the RU consumes by staying active
RSRP_THRESHOLD_DBM: float = -85
GRAPHICAL_SCALING_FACTOR: float = 0.85
RU_SIGNAL_STRENGTH: float = 36.98 #dBm
DU_DISTANCE_FROM_CENTER: float = 500
RU_DISTANCE_FROM_DU: float = 250

# SETTINGS
SHOW_EXPERIMENT_STATS = True

# RENDERING INFO
UE_IMAGE: str = "images/ue.gif"
RU_IMAGE: str = "images/ru.gif"
RU_OFF_IMAGE: str = "images/ru_off.gif"
DU_IMAGE: str = "images/du.gif"
DU_OFF_IMAGE: str = "images/du_off.gif"

# CLASSES

class Point:
    def __init__(self,x: float,y: float)->None:
        self.x = x
        self.y = y

    def __add__(self,other)->'Point':
        return Point(other.x+self.x,other.y+self.y)

    def dist(self,otherPoint)-> float: 
        # Returns Euclidean distance between this point and another point.
        return sqrt(pow(self.x-otherPoint.x,2) + pow(self.y-otherPoint.y,2))

class O_RU:
    def __init__(self, p: Point)->None:
        self.p: Point = p
        self.active = True
        self.connectedUEs: set = set()
        self.connectedDU: O_DU | None = None
        self.height = 10 #meters

        self.transmissionPower: float = RU_SIGNAL_STRENGTH
        self.operatingFrequency: float = 3.7 #GHz, N77 Freq Band
        self.numberOfTransmissionAntennas: int = 8
        self.codeRate = 0.753
        self.modulationOrder = 6
        self.MIMOLayers = 4
        
        #self.fieldOFDM: matrix = matrix(fromfunction(lambda i, j: ), ())

        self.initializeTurtle()

    def initializeTurtle(self)->None:
        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.p.x/GRAPHICAL_SCALING_FACTOR,self.p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(RU_IMAGE)
        self.turtle.setheading(90)

    def getPosition(self)->Point:
        return self.p
    
    def getConnectedUEs(self):
        return self.connectedUEs
    
    def connectUE(self,UE)->None:
        self.connectedUEs.add(UE)

    def removeUE(self,UE)->None:
        self.connectedUEs.remove(UE)

    def connectDU(self,DU)->None:
        if self.connectedDU:
            self.connectedDU.removeRU(self)

        DU.connectRU(self)
        self.connectedDU = DU

    def getDU(self)->'O_DU':
        return self.connectedDU # type: ignore
    
    def sleep(self)->None:
        self.turtle.shape(RU_OFF_IMAGE)
        self.active = False

    def wake(self)->None:
        self.turtle.shape(RU_IMAGE)
        self.active = True
        
    def status(self):
        return self.active
        
    def getProcessingLoad(self)->float:
        GOPS = 0
        for ue in self.getConnectedUEs():
            GOPS += 0.4*(3*ue.getRU().numberOfTransmissionAntennas + ue.getRU().numberOfTransmissionAntennas**2 + ue.getRU().modulationOrder*ue.getRU().codeRate*ue.getRU().MIMOLayers/3)/5
        self.processingLoad = GOPS/1600
        if self.processingLoad > 1:
            self.processingLoad = 1
        return self.processingLoad


class O_DU:
    def __init__(self, p: Point)->None:
        self.p = p
        self.connectedRUs: set = set()
        self.active = True
        
        self.processingLoad = 0 #percent

        self.initializeTurtle()

    def initializeTurtle(self):
        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.p.x/GRAPHICAL_SCALING_FACTOR,self.p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(DU_IMAGE)
        self.turtle.setheading(90)

    def connectRU(self, RU: O_RU)->None:
        self.connectedRUs.add(RU)

    def removeRU(self, RU: O_RU)->None:
        self.connectedRUs.remove(RU)
        
    def getConnectedRUs(self)->set:
        return self.connectedRUs
    
    def getConnectedUEs(self)->list:
        T = []
        for unit in self.getConnectedRUs():
            
            for ue in unit.getConnectedUEs():
                T.append(ue)
        return T

    def getPosition(self)->Point:
        return self.p
    
    def sleep(self)->None:
        self.turtle.shape(DU_OFF_IMAGE)
        self.active = False

    def wake(self)->None:
        self.turtle.shape(DU_IMAGE)
        self.active = True
    
    def getProcessingLoad(self)->float:
        # This function generates GOPS according to the paper "Dynamic Placement of O-CU and O-DU Functionalities in Open-RAN Architecture" by Hojeij et al.
        GOPS = 0
        for ue in self.getConnectedUEs():
            GOPS += 0.5*(3*ue.getRU().numberOfTransmissionAntennas + ue.getRU().numberOfTransmissionAntennas**2 + ue.getRU().modulationOrder*ue.getRU().codeRate*ue.getRU().MIMOLayers/3)/5
        self.processingLoad = GOPS/1600
        if self.processingLoad > 1:
            self.processingLoad = 1
        return self.processingLoad

    def status(self):
        return self.active

class UE:
    def __init__(self, p: Point)->None:
        self.p: Point = p
        self.RU = None
        self.freq: int = 3300 # MHz

        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(p.x/GRAPHICAL_SCALING_FACTOR,p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(UE_IMAGE)
        self.turtle.setheading(90)

    def getPosition(self)->Point:
        return self.p
    
    def walk(self, d: Point)->None:
        self.p += d
        self.turtle.setposition(self.p.x/GRAPHICAL_SCALING_FACTOR,self.p.y/GRAPHICAL_SCALING_FACTOR)
    
    def detachFromRU(self)->None:
        if self.RU:
            self.RU.removeUE(self)
            self.RU = None
    
    def attachToRU(self,RU)->None:
        self.detachFromRU()
    
        self.RU = RU
        RU.connectUE(self)

    def getRU(self)->O_RU:
        return self.RU # type: ignore

class networkSimulation:
    def __init__(self, n: int, m: int, k: int, s: float, dt=0.1)->None:
        self.numRUs = n
        self.RUs = {}

        self.numDUs = m
        self.DUs = {}

        self.numUEs = k
        self.UEs = {}

        self.simulationSideLength = s # in meters
        self.timeStepLength = dt # amount of time one frame goes for
        self.totalEnergyConsumption = 0 # in watts

        self.screen = Screen()
        self.screen.setup(width=1500,height=1500)
        self.screen.title("Stochastic DQN Model for Joint Optimization of Delay and Energy Efficiency Simulation")
        self.screen.tracer(0)

        self.screen.register_shape(UE_IMAGE)
        self.screen.register_shape(RU_IMAGE)
        self.screen.register_shape(DU_IMAGE)
        self.screen.register_shape(RU_OFF_IMAGE)
        self.screen.register_shape(DU_OFF_IMAGE)

        self.UEConnectionTurtle = Turtle()
        self.UEConnectionTurtle.speed(0)
        self.UEConnectionTurtle.penup()
        self.UEConnectionTurtle.pencolor("blue")
        self.UEConnectionTurtle.hideturtle()

        self.RUDUConnectionTurtle = Turtle()
        self.RUDUConnectionTurtle.speed(0)
        self.RUDUConnectionTurtle.penup()
        self.RUDUConnectionTurtle.pencolor("green")
        self.RUDUConnectionTurtle.hideturtle()

        self.SimulationStatisticsTurtle = Turtle()
        self.SimulationStatisticsTurtle.speed(0)
        self.SimulationStatisticsTurtle.penup()
        self.SimulationStatisticsTurtle.hideturtle()
        
    def generateChannelQualityMatrix(self) -> matrix:
        mathcalH = fromfunction(vectorize(lambda i,j: rssi(self.RUs[i],self.UEs[j])) ,(self.numRUs*self.numDUs, self.numUEs), dtype=float )
        return mathcalH
    
    def generateGeoLocationMatrix(self) -> matrix:
        G = fromfunction(vectorize(lambda i,j: self.UEs[i].getPosition().x if j == 0 else self.UEs[i].getPosition().y), (self.numUEs,2), dtype=float)
        return G
    
    def generateConnectionQualityVector(self) -> matrix:
        mathcalH = self.generateChannelQualityMatrix()
        G = self.generateGeoLocationMatrix()
        mathcalV = []
        for i in range(self.numUEs):
            mathcalV.append(outer(G[i,:],mathcalH[:,i]))
        return matrix(mathcalV)
    
    def generateDelayMatrix(self) -> matrix:
        mathcalP = fromfunction(vectorize(lambda i,j: self.RUs[j].getPosition().dist(self.DUs[i].getPosition())/c + self.DUs[i].getProcessingLoad()*0.035 + 0.4 * rng.uniform(0.025,0.25)), (self.numDUs, self.numRUs*self.numDUs), dtype=float)
        return mathcalP
    
    def generateProcessingLoadVector(self) -> matrix:
        mathcalZ = [self.DUs[unit].updateProcessingLoad() for unit in self.DUs]
        return matrix(mathcalZ)
    
    def generateStateVector(self) -> matrix:
        return matrix([self.generateDelayMatrix(), self.generateConnectionQualityVector(), self.generateProcessingLoadVector()])
    
    def updateTotalEnergyConsumption(self) -> None:
        # The DUs in this simulation are based on a generic 2nd Gen Intel Xeon processor
        
        E_DU_idle = 90 #watts
        E_DU_max = 650 #watts
        for unit in self.DUs.values():
            if unit.status():
                self.totalEnergyConsumption += (unit.getProcessingLoad()*(E_DU_max + E_DU_idle) + E_DU_idle)*(1/3600)
                
        E_RU_idle = 80 #watts
        E_RU_max = 120 #watts
                
        for unit in self.RUs.values():
            if unit.status():
                self.totalEnergyConsumption += (unit.getProcessingLoad()*(E_RU_max + E_RU_idle) + E_RU_idle)*(1/3600)
        
            
    def getTotalEnergyConsumption(self):
        return self.totalEnergyConsumption

    def updateStatisticsDisplay(self, _: int):

        X_POSITION = -1*self.screen.window_width()//2+50
        Y_POSITION = self.screen.window_height()//2-50

        self.SimulationStatisticsTurtle.goto(X_POSITION, Y_POSITION)
        self.SimulationStatisticsTurtle.write(f"Time: {_} second(s)", align="left", font=("Arial", 16, "normal"))

        self.SimulationStatisticsTurtle.goto(X_POSITION, Y_POSITION - 25)
        self.SimulationStatisticsTurtle.write(f"O-RUs: {self.numRUs*self.numDUs}", align="left", font=("Arial", 16, "normal"))

        self.SimulationStatisticsTurtle.goto(X_POSITION, Y_POSITION - 50)
        self.SimulationStatisticsTurtle.write(f"O-DUs: {self.numDUs}", align="left", font=("Arial", 16, "normal"))

        self.SimulationStatisticsTurtle.goto(X_POSITION, Y_POSITION - 75)
        self.SimulationStatisticsTurtle.write(f"UEs: {self.numUEs}", align="left", font=("Arial", 16, "normal"))
        
        self.SimulationStatisticsTurtle.goto(X_POSITION, Y_POSITION - 100)
        self.SimulationStatisticsTurtle.write(f"Energy Consumption: {round(self.getTotalEnergyConsumption()/1e+3,4)} kWh", align="left", font=("Arial", 16, "normal"))
        
        #self.SimulationStatisticsTurtle.goto(X_POSITION, Y_POSITION - 125)
        #self.SimulationStatisticsTurtle.write(f"Channel Quality Matrix", align="left", font=("Arial", 16, "normal"))
        # Here we are showing the channel quality matrix \mathcal{H} (more in README.md)
        #mathcalH = self.generateChannelQualityMatrix()
        #for row in range(self.numRUs*self.numDUs):
        #    self.SimulationStatisticsTurtle.goto(X_POSITION, Y_POSITION - (150 + 25*row))
        #    self.SimulationStatisticsTurtle.write(f"{' '.join(str(round(x,2)) for x in mathcalH[row])}", align="left", font=("Arial", 16, "normal"))

    def updateComponentConnectionDisplay(self):
        for unit in self.RUs.values():
            self.RUDUConnectionTurtle.penup()
            if unit.getDU():
                self.RUDUConnectionTurtle.goto(unit.getDU().getPosition().x/GRAPHICAL_SCALING_FACTOR, unit.getDU().getPosition().y/GRAPHICAL_SCALING_FACTOR)
                self.RUDUConnectionTurtle.pendown()
                self.RUDUConnectionTurtle.goto(unit.getPosition().x/GRAPHICAL_SCALING_FACTOR,unit.getPosition().y/GRAPHICAL_SCALING_FACTOR)

        for ue in self.UEs.values() :
            if ue.getRU():
                self.UEConnectionTurtle.goto(ue.getPosition().x/GRAPHICAL_SCALING_FACTOR, ue.getPosition().y/GRAPHICAL_SCALING_FACTOR)
                self.UEConnectionTurtle.pendown()
                self.UEConnectionTurtle.goto(ue.getRU().getPosition().x/GRAPHICAL_SCALING_FACTOR, ue.getRU().getPosition().y/GRAPHICAL_SCALING_FACTOR)
                self.UEConnectionTurtle.penup()
            
    def initializeComponents(self):
        self.RUs = {}
        self.DUs = {}
        self.UEs = {}
        for du in range(self.numDUs):
            # Create m DUs, assign IDs to them
            # Place the DUs automatically
            D_THETA = rad2deg(pi*du/self.numDUs)
            newDU = O_DU(Point(DU_DISTANCE_FROM_CENTER*cos(D_THETA),DU_DISTANCE_FROM_CENTER*sin(D_THETA)))
            self.DUs[du] = newDU
            for ru in range(self.numRUs):
                R_THETA = rad2deg(2*pi*ru/self.numRUs)
                newRU = O_RU(Point(RU_DISTANCE_FROM_DU*cos(R_THETA) + newDU.getPosition().x,RU_DISTANCE_FROM_DU*sin(R_THETA) + newDU.getPosition().y))
                self.RUs[len(self.RUs)] = newRU
                newRU.connectDU(newDU)

        for id in range(self.numUEs):
            # Create k UEs, assign IDs to them.
            newUE = UE(createRandomPoint(self.simulationSideLength/2))
            self.UEs[id] = newUE

    def run(self, simulationLength: int)->None:
        self.initializeComponents()
        
        

        # Main loop
        for _ in range(0,simulationLength):
            time.sleep(self.timeStepLength)
            self.UEConnectionTurtle.clear()
            self.RUDUConnectionTurtle.clear()
            self.SimulationStatisticsTurtle.clear()
                        
            if SHOW_EXPERIMENT_STATS:
                self.updateStatisticsDisplay(_)
                
            self.updateComponentConnectionDisplay()
            
            
            # Random Walk for UEs
            for ue in self.UEs.values():
                ue.walk(createRandomPoint(8))
                bestConnectedRU = ue.getRU() if ue.getRU() else self.RUs[0]
                bestConnectedRURSRP = rsrp(bestConnectedRU,ue)
                
                for unit in self.RUs.values():
                    if unit.active:
                        tentativeRSRP = rsrp(unit,ue)
                        #print(tentativeRSRP)
                        if tentativeRSRP >= RSRP_THRESHOLD_DBM:
                            bestConnectedRU = unit
                            bestConnectedRURSRP = rsrp(bestConnectedRU,ue)
                                
                if bestConnectedRURSRP < RSRP_THRESHOLD_DBM:
                    ue.detachFromRU()
                else:
                    ue.attachToRU(bestConnectedRU)
            self.updateTotalEnergyConsumption()
            self.screen.update()

# FUNCTIONS

def calculatePathLoss(ru: O_RU, ue: UE)->float:
    #UMi path loss based on https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.00.00_60/tr_138901v180000p.pdf
    
    hprime_bs = ru.height - 1.5
    
    d_2d = ru.getPosition().dist(ue.getPosition())
    d_3d = sqrt(pow((ru.getPosition().x-ue.getPosition().x),2) + pow((ru.getPosition().y-ue.getPosition().y),2) + pow((ru.height),2))
    d_bp = 4*hprime_bs*1.5*ru.operatingFrequency/c
    
    
    pl1 = 32.4 + 21*log10(d_3d)+20*log10(ru.operatingFrequency)
    pl2 = 32.4 + 40*log10(d_3d)+20*log10(ru.operatingFrequency)-9.5*log10(d_bp**2+(ru.height-1.5)**2)
    
    return pl1 if d_2d > 10 and d_2d < d_bp else pl2

def rsrp(ru: O_RU, ue: UE)->float:
    # Calculation based on "Realistic Signal Strength Simulation for ORAN Testing Environments" by Nour Bahtite
    return ru.transmissionPower - calculatePathLoss(ru, ue)

def rssi(ru: O_RU, ue: UE)->float:
    return rsrp(ru, ue) + log10(12*275)

def createRandomPoint(s)->Point:
    return Point(rng.uniform(-1*s,s),rng.uniform(-1*s,s))