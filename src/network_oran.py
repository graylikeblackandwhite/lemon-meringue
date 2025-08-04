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
        return sqrt((self.x**2-otherPoint.x**2) + (self.y**2-otherPoint.y**2))

class O_RU:
    def __init__(self, p: Point)->None:
        self.p = p
        self.active = True

        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(p.x,p.y)
        self.turtle.shape(RU_IMAGE)
        self.turtle.setheading(90)

    def getPosition(self)->Point:
        return self.p
    


class O_DU:
    def __init__(self, p: Point)->None:
        self.p = p
        self.connectedRUs = []
        self.active = True

        self.turtle = Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(p.x,p.y)
        self.turtle.shape(DU_IMAGE)
        self.turtle.setheading(90)

    def addRU(self, RU: O_RU)->None:
        # Don't use this function! A better function which removes the RU from its previous DU can be found in this file.
        self.connectedRUs.append(RU)

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
        self.turtle.setposition(p.x,p.y)
        self.turtle.shape(UE_IMAGE)
        self.turtle.setheading(90)

    def getPosition(self)->Point:
        return self.p
    
    def walk(self, d: Point)->None:
        self.p += d
        self.turtle.setposition(self.p.x,self.p.y)
    
    def attachToRU(self,RU)->None:
        self.RU = RU

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


        screen.register_shape(UE_IMAGE)
        screen.register_shape(RU_IMAGE)
        screen.register_shape(DU_IMAGE)

        turtleRUs = {}
        turtleDUs = {}
        turtleUEs = {}

        totalEnergyConsumption = 0

        for id in range(self.numRUs-1):
            # Create n RUs, assign IDs to them.
            newRU = O_RU(createRandomPoint(self.simulationSideLength/2))
            self.RUs[id] = newRU

        for id in range(self.numDUs-1):
            # Create m DUs, assign IDs to them.
            
            # Place the DUs automatically
            D_THETA = rad2deg((pi*id)/(2*self.numDUs))
            L = rng.uniform(0, self.simulationSideLength*sqrt(2)/2)
            newDU = O_DU(Point(L*cos(D_THETA),L*sin(D_THETA)))
            self.DUs[id] = newDU

            newTurtleDU = Turtle()
            newTurtleDU.penup()
            newTurtleDU.speed(0)
            newTurtleDU.shape(DU_IMAGE)
            newTurtleDU.setheading(90)

        for id in range(self.numUEs-1):
            # Create k UEs, assign IDs to them.
            newUE = UE(createRandomPoint(self.simulationSideLength/2))
            self.UEs[id] = newUE

            newTurtleUE = Turtle()
            newTurtleUE.penup()
            newTurtleUE.speed(0)
            newTurtleUE.shape(UE_IMAGE)
            newTurtleUE.setheading(90)

        print(self.RUs)
        print(self.DUs)
        print(self.UEs)

        for _ in range(0,simulationLength):
            for id in range(self.numDUs-1):
                if self.DUs[id].active:
                    totalEnergyConsumption += E_S_DU*self.timeStepLength
            
            for id in range(self.numRUs-1):
                if self.RUs[id].active:
                    totalEnergyConsumption += E_S_RU*self.timeStepLength

            # Random Walk for UEs
            for id in range(self.numUEs-1):
                self.UEs[id].walk(createRandomPoint(8))


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

