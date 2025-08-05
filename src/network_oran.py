from numpy import * # pyright: ignore[reportWildcardImportFromLibrary]
from turtle import * # pyright: ignore[reportWildcardImportFromLibrary]
from enum import Enum
import time


rng = random.default_rng()

# CONSTANTS

c = 299792458 # speed of light
P_0 = 30.0 # in dBm
E_S_DU = 2 # megajoules, how much energy the DU consumes by staying active
E_S_RU = 1 # megajoules, how much energy the RU consumes by staying active
RU_CELL_RADIUS = 100 # meters
GRAPHICAL_SCALING_FACTOR = 0.85

# RENDERING INFO
UE_IMAGE = "images/ue.gif"
RU_IMAGE = "images/ru.gif"
DU_IMAGE = "images/du.gif"

# CLASSES

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __add__(self,other):
        return Point(other.x+self.x,other.y+self.y)

    def dist(self,otherPoint)-> float: 
        # Returns Euclidean distance between this point and another point.
        return sqrt(pow(self.x-otherPoint.x,2) + pow(self.y-otherPoint.y,2))

class O_RU:
    def __init__(self, p: Point)->None:
        self.p = p
        self.active = True
        self.connectedUEs = set()
        self.connectedDU = None

        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(p.x/GRAPHICAL_SCALING_FACTOR,p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(RU_IMAGE)
        self.turtle.setheading(90)

    def getPosition(self)->Point:
        return self.p
    
    def getConnectedUEs(self):
        return self.connectedUEs
    
    def addUE(self,UE)->None:
        self.connectedUEs.add(UE)

    def removeUE(self,UE)->None:
        self.connectedUEs.remove(UE)

    def connectDU(self,DU)->None:
        if self.connectedDU:
            self.connectedDU.removeRU(self)
            self.connectedDU = None

        self.connectedDU = DU

    def getDU(self):
        return self.connectedDU


class O_DU:
    def __init__(self, p: Point)->None:
        self.p = p
        self.connectedRUs = set()
        self.active = True

        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(p.x/GRAPHICAL_SCALING_FACTOR,p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(DU_IMAGE)
        self.turtle.setheading(90)

    def connectRU(self, RU: O_RU)->None:
        self.connectedRUs.add(RU)

    def removeRU(self, RU: O_RU)->None:
        self.connectedRUs.remove(RU)

    def getPosition(self)->Point:
        return self.p

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
    
    def attachToRU(self,RU)->None:
        if self.RU:
            self.RU.removeUE(self)
            self.RU = None
    
        self.RU = RU
        RU.addUE(self)

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

    def assignRUtoDU(self,RU,DU)->None:
        pass

    def run(self, simulationLength: int)->None:
        # Reset components in simulation
        self.RUs = {}
        self.DUs = {}
        self.UEs = {}

        # Set up rendering
        screen = Screen()
        screen.setup(width=1500,height=1500)
        screen.title("DQN Model for Joint Optimization of Delay and Energy Efficiency Simulation")
        screen.tracer(0)

        screen.register_shape(UE_IMAGE)
        screen.register_shape(RU_IMAGE)
        screen.register_shape(DU_IMAGE)

        UEConnectionTurtle = Turtle()
        UEConnectionTurtle.speed(0)
        UEConnectionTurtle.penup()
        UEConnectionTurtle.pencolor("blue")
        UEConnectionTurtle.hideturtle()

        RUDUConnectionTurtle = Turtle()
        RUDUConnectionTurtle.speed(0)
        RUDUConnectionTurtle.penup()
        RUDUConnectionTurtle.pencolor("green")
        RUDUConnectionTurtle.hideturtle()

        totalEnergyConsumption = 0

        for id in range(self.numDUs-1):
            # Create m DUs, assign IDs to them.
            
            # Place the DUs automatically
            D_THETA = rad2deg((pi*id)/(2*self.numDUs))
            L = rng.uniform(20, self.simulationSideLength*sqrt(2)/2)
            newDU = O_DU(Point(L*cos(D_THETA),L*sin(D_THETA)))
            self.DUs[id] = newDU

        for id in range(self.numRUs-1):
            # Create n RUs, assign IDs to them.
            newRU = O_RU(createRandomPoint(self.simulationSideLength/2))
            self.RUs[id] = newRU

            closestActiveDU = self.DUs[0]
            closestActiveDUDist = closestActiveDU.getPosition().dist(self.RUs[id].getPosition())
            for unit in self.DUs.values():
                tentativeDU = self.RUs[id].getPosition().dist(unit.getPosition())
                if tentativeDU < closestActiveDUDist and unit.active == True:
                    closestActiveDU = unit
                    closestActiveDUDist = closestActiveDU.getPosition().dist(self.RUs[id].getPosition())
            
            self.RUs[id].connectDU(closestActiveDU)


        for id in range(self.numUEs-1):
            # Create k UEs, assign IDs to them.
            newUE = UE(createRandomPoint(self.simulationSideLength/2))
            self.UEs[id] = newUE

        for _ in range(0,simulationLength):
            UEConnectionTurtle.clear()
            RUDUConnectionTurtle.clear()

            for unit in self.RUs.values():
                RUDUConnectionTurtle.penup()
                if unit.getDU():
                    RUDUConnectionTurtle.goto(unit.getDU().getPosition().x/GRAPHICAL_SCALING_FACTOR, unit.getDU().getPosition().y/GRAPHICAL_SCALING_FACTOR)
                    RUDUConnectionTurtle.pendown()
                    RUDUConnectionTurtle.goto(unit.getPosition().x/GRAPHICAL_SCALING_FACTOR,unit.getPosition().y/GRAPHICAL_SCALING_FACTOR)


            

            # Random Walk for UEs
            for ue in self.UEs.values():
                ue.walk(createRandomPoint(8))
                # Heuristic: Connect UE to nearest RU
                # TODO: Base the heuristic on RSS, not distance.
                closestActiveRU = self.RUs[0]
                closestActiveRUDist = fspl(closestActiveRU.getPosition().dist(ue.getPosition()), 3300)
                for unit in self.RUs.values():
                    print(fspl(ue.getPosition().dist(unit.getPosition()),3300))
                    if fspl(ue.getPosition().dist(unit.getPosition()),3300) > -23:
                        tentativeRU = fspl(ue.getPosition().dist(unit.getPosition()),3300)
                        if tentativeRU < closestActiveRUDist and unit.active == True:
                            closestActiveRU = unit
                            closestActiveRUDist = fspl(closestActiveRU.getPosition().dist(ue.getPosition()),3300)
                
                if fspl(ue.getPosition().dist(closestActiveRU.getPosition()),3300) <= -23:
                    closestActiveRU = None
                else:
                    ue.attachToRU(closestActiveRU)
                    UEConnectionTurtle.goto(ue.getPosition().x/GRAPHICAL_SCALING_FACTOR, ue.getPosition().y/GRAPHICAL_SCALING_FACTOR)
                    UEConnectionTurtle.pendown()
                    UEConnectionTurtle.goto(ue.RU.getPosition().x/GRAPHICAL_SCALING_FACTOR, ue.RU.getPosition().y/GRAPHICAL_SCALING_FACTOR)
                    UEConnectionTurtle.penup()

                print("#####")
            screen.update()
            time.sleep(self.timeStepLength)

        print(totalEnergyConsumption)

        

# FUNCTIONS

def fspl(d, f)->float:
    return 20*log10(d)+20*log10(f)+20*log10(4*pi/c)

def rss(d,f)->float:
    # Returns Received Signal Strength given a distance in meters and a frequency in Gigahertz.
    return P_0-20*log10((4*pi*d)/(c/f))

def createRandomPoint(s)->Point:
    # Creates a random point from -s to s
    return Point(rng.uniform(-1*s,s),rng.uniform(-1*s,s))

