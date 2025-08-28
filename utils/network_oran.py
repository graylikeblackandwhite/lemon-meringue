#   Simulation software implementation
#   Written by Gray
#   TODO: Implement custom movement paths for devices
#   We can start from an environment with static UEs that don't change association.
#   We can implement dynamic traffic demands into the model to make it more realistic.
#   Measure energy consumption using default solution (no change), we can do this side-by-side in the simulation. One side can have the model activated, the other can have no model active, we can compare the energy consumption between both scenarios.
#   Even if there's no srsRAN, *some* conferences do accept these kinds of ad hoc simulations. If the math is solid we can submit to a conference.
#   If the solution doesn't show promising results, we can find a different scope for the paper.

from collections import deque
from datetime import datetime
from enum import Enum
from itertools import product
import numpy as np
import time
import turtle
import pandas as pd
import torch
import os

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

initial_cwd = os.getcwd()
print(f"Initial CWD: {initial_cwd}")

# Change the working directory
new_directory = f"{get_project_root()}/utils/" # Replace with your desired path
# You might need to create the directory if it doesn't exist
os.makedirs(new_directory, exist_ok=True) 
os.chdir(new_directory)

# Get the updated current working directory
updated_cwd = os.getcwd()
print(f"Updated CWD: {updated_cwd}")

seed: float = 427
rng: np.random.Generator = np.random.default_rng(seed=seed)

# CONSTANTS

c: float = 299792458 # speed of light
RSRP_THRESHOLD_DBM: float = -92 #dBm
RSRP_REWARD_THRESHOLD_DBM: float = -80 #dBm
GRAPHICAL_SCALING_FACTOR: float = 0.85
RU_SIGNAL_STRENGTH: float = 36.98 #dBm
DU_DISTANCE_FROM_CENTER: float = 500 #meters
RU_DISTANCE_FROM_DU: float = 250 #meters

RU_CAPACITY_THRESHOLD: int = 20 # Number of UEs that can be connected to a RU at once
DU_CAPACITY_THRESHOLD: int = 6 # Number of UEs that can be connected to a DU at once
# SETTINGS
SHOW_EXPERIMENT_STATS: bool = True

# RENDERING INFO
UE_IMAGE: str = "images/ue.gif"
RU_IMAGE: str = "images/ru.gif"
RU_OFF_IMAGE: str = "images/ru_off.gif"
DU_IMAGE: str = "images/du.gif"
DU_OFF_IMAGE: str = "images/du_off.gif"

# ENUMS
class NetworkSimulationActionType(Enum):
    SWITCH = 0
    RU_SLEEP = 1
    DU_SLEEP = 2

# CLASSES

class Point:
    def __init__(self,x: float,y: float)->None:
        self.x = x
        self.y = y

    def __add__(self,other)->'Point':
        return Point(other.x+self.x,other.y+self.y)

    def dist(self,other_point)-> float: 
        # Returns Euclidean distance between this point and another point.
        return np.sqrt(pow(self.x-other_point.x,2) + pow(self.y-other_point.y,2))

class O_RU:
    def __init__(self, p: Point, show_graphics=False)->None:
        self.p: Point = p
        self.active = True
        self.connected_ues: set = set()
        self.connected_du: O_DU | None = None
        self.height = 10 #meters

        self.transmission_power: float = RU_SIGNAL_STRENGTH
        self.operating_frequency: float = 3.7 #GHz, N77 Freq Band
        self.number_of_transmission_antennas: int = 8
        self.code_rate = 0.753
        self.modulation_order = 6
        self.MIMO_layers = 4
        
        #self.fieldOFDM: matrix = matrix(fromfunction(lambda i, j: ), ())

        if show_graphics:
            self.initialize_turtle()
        self.graphics = show_graphics
        
    def __del__(self)->None:
        if self.graphics:
            del self.turtle

    def initialize_turtle(self)->None:
        self.turtle = turtle.Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.p.x/GRAPHICAL_SCALING_FACTOR,self.p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(RU_IMAGE)
        self.turtle.setheading(90)

    def get_position(self)->Point:
        return self.p
    
    def get_connected_ues(self):
        return self.connected_ues
    
    def connect_ue(self,UE)->None:
        self.connected_ues.add(UE)

    def remove_ue(self,UE)->None:
        self.connected_ues.remove(UE)

    def connect_du(self,DU)->None:
        if self.connected_du:
            self.connected_du.remove_ru(self)
            self.connected_du = None
        DU.connect_ru(self)
        self.connected_du = DU

    def get_du(self)->'O_DU':
        return self.connected_du # type: ignore

    def sleep(self)->None:
        if self.graphics:
            self.turtle.shape(RU_OFF_IMAGE)
        self.active = False

        ue: UE
        for ue in self.get_connected_ues().copy():
            ue.detach_from_ru()

    def wake(self)->None:
        if self.graphics:
            self.turtle.shape(RU_IMAGE)
        self.active = True
        
    def status(self)->bool:
        return self.active
        
    def get_processing_load(self)->float:
        GOPS = 0
        for ue in self.get_connected_ues():
            GOPS += 0.4*(3*ue.get_ru().number_of_transmission_antennas + ue.get_ru().number_of_transmission_antennas**2 + ue.get_ru().modulation_order*ue.get_ru().code_rate*ue.get_ru().MIMO_layers/3)/5
        self.processing_load = max(1,GOPS/1600)
        return self.processing_load


class O_DU:
    def __init__(self, p: Point, show_graphics=False)->None:
        self.p = p
        self.connected_rus: set = set()
        self.active = True
        
        self.processing_load = 0 #percent

        if show_graphics:
            self.initialize_turtle()
        self.graphics = show_graphics

    def initialize_turtle(self):
        self.turtle = turtle.Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.p.x/GRAPHICAL_SCALING_FACTOR,self.p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(DU_IMAGE)
        self.turtle.setheading(90)

    def connect_ru(self, RU: O_RU)->None:
        self.connected_rus.add(RU)

    def remove_ru(self, RU: O_RU)->None:
        self.connected_rus.remove(RU)

    def get_connected_rus(self)->set:
        return self.connected_rus

    def get_connected_ues(self)->set:
        T = set()
        for unit in self.get_connected_rus():
            for ue in unit.get_connected_ues():
                T.add(ue)
        return T

    def get_position(self)->Point:
        return self.p
    
    def sleep(self)->None:
        if self.graphics:
            self.turtle.shape(DU_OFF_IMAGE)
        self.active = False

    def wake(self)->None:
        if self.graphics:
            self.turtle.shape(DU_IMAGE)
        self.active = True

    def get_processing_load(self)->float:
        # This function generates GOPS according to the paper "Dynamic Placement of O-CU and O-DU Functionalities in Open-RAN Architecture" by Hojeij et al.
        GOPS = 0
        for ue in self.get_connected_ues():
            GOPS += 0.5*(3*ue.get_ru().number_of_transmission_antennas + ue.get_ru().number_of_transmission_antennas**2 + ue.get_ru().modulation_order*ue.get_ru().code_rate*ue.get_ru().MIMO_layers/3)/5
        self.processing_load = max(1, GOPS/1600)
        return self.processing_load

    def status(self)->bool:
        return self.active

class UE:
    def __init__(self, p: Point, show_graphics=False)->None:
        self.p: Point = p
        self.RU: O_RU | None = None
        self.freq: int = 3300 # MHz
        if show_graphics:
            self.initialize_turtle()
        self.graphics = show_graphics

    def initialize_turtle(self)->None:
        self.turtle = turtle.Turtle()
        self.turtle.penup()
        self.turtle.speed(0)
        self.turtle.setposition(self.p.x/GRAPHICAL_SCALING_FACTOR,self.p.y/GRAPHICAL_SCALING_FACTOR)
        self.turtle.shape(UE_IMAGE)
        self.turtle.setheading(90)

    def get_position(self)->Point:
        return self.p
    
    def walk(self, d: Point)->None:
        self.p += d
        
        if self.graphics:
            self.turtle.setposition(self.p.x/GRAPHICAL_SCALING_FACTOR,self.p.y/GRAPHICAL_SCALING_FACTOR)
    
    def detach_from_ru(self)->None:
        if self.RU:
            self.RU.remove_ue(self)
            self.RU = None

    def attach_to_ru(self,RU)->None:
        self.detach_from_ru()

        self.RU = RU
        RU.connect_ue(self)

    def get_ru(self)->O_RU:
        return self.RU # type: ignore

class NetworkSimulation:
    def __init__(self, n: int, m: int, k: int, s: float, show_graphics: bool = False, dt=0.1, seed=42)->None:
        self.main_loop_step = -1
        self.running = False
        
        self.num_rus = n
        self.rus = {}

        self.num_dus = m
        self.dus = {}

        self.num_ues = k
        self.ues = {}
        
        self.alpha = 1.2
        self.beta = 0.4
        self.seed = 42

        self.simulation_side_length = s # in meters
        self.time_step_length = dt # amount of time one frame goes for
        self.total_energy_consumption = 0 # in watts
        
        self.graphics = show_graphics

    def generate_delay_matrix(self)->torch.Tensor:
        delay_matrix: torch.Tensor = torch.zeros((self.num_rus,self.num_dus))
        ru: int
        du: int
        for ru in range(self.num_rus):
            for du in range(self.num_dus):
                if len(self.rus[ru].get_connected_ues()) != 0:
                    delay_matrix[ru][du] = calculate_fronthaul_delay(self.rus[ru],self.dus[du])*(sum([rsrp(self.rus[ru], ue) for ue in self.rus[ru].get_connected_ues()])/len(self.rus[ru].get_connected_ues()))
        return delay_matrix

    def generate_state_vector(self) -> torch.Tensor:
        return torch.flatten(self.generate_delay_matrix())
    
    def calculate_ru_power_reward(self) -> float:
        r = 0
        ue: UE
        for ue in self.ues.values():
            if ue.get_ru():
                if rsrp(ue.get_ru(),ue) < RSRP_REWARD_THRESHOLD_DBM:
                    r += 1
                else:
                    r -= 2
        return r
    
    def calculate_ru_capacity_reward(self) -> float:
        return sum([1 if len(ru.get_connected_ues()) <= RU_CAPACITY_THRESHOLD else -2.0 for ru in self.rus.values()])
    
    def calculate_du_capacity_reward(self) -> float:
        return sum([1 if len(du.get_connected_ues()) <= DU_CAPACITY_THRESHOLD else -2.0 for du in self.dus.values()])

    def calculate_sleep_reward(self) -> float:
        return (len([unit for unit in self.rus.values() if unit.status()]) + len([unit for unit in self.dus.values() if unit.status()]))
    
    def calculate_normalizer_term(self) -> float:
        num_ru_sleep = len([unit for unit in self.rus.values() if not unit.status()])
        num_du_sleep = len([unit for unit in self.dus.values() if not unit.status()])
        return (self.num_ues + self.num_dus + num_ru_sleep + num_du_sleep)/((1-c)*self.num_rus)
    
    def calculate_reward(self) -> torch.Tensor: 
        return torch.tensor(((self.calculate_ru_power_reward() + self.calculate_sleep_reward() + self.calculate_du_capacity_reward() + self.calculate_ru_capacity_reward())*self.alpha - self.beta*self.calculate_normalizer_term()*self.calculate_average_fronthaul_delay())/2*self.calculate_normalizer_term(), dtype=torch.float32)

    def update_total_energy_consumption(self) -> None:
        # The DUs in this simulation are based on a generic 2nd Gen Intel Xeon processor
        
        E_DU_idle = 90 #watts
        E_DU_max = 650 #watts
        du: O_DU
        for du in self.dus.values():
            if du.status():
                self.total_energy_consumption += (du.get_processing_load()*(E_DU_max + E_DU_idle) + E_DU_idle)*(1/3600)

        E_RU_idle = 80 #watts
        E_RU_max = 120 #watts
        
        ru: O_RU
        for ru in self.rus.values():
            if ru.status():
                self.total_energy_consumption += (ru.get_processing_load()*(E_RU_max + E_RU_idle) + E_RU_idle)*(1/3600)

    def get_total_energy_consumption(self)->float:
        return self.total_energy_consumption

    def update_statistics_display(self, _: int):

        X_POSITION = -1*self.screen.window_width()//2+50
        Y_POSITION = self.screen.window_height()//2-50

        self.simulation_statistics_turtle.goto(X_POSITION, Y_POSITION)
        self.simulation_statistics_turtle.write(f"Time: {_} second(s)", align="left", font=("Arial", 16, "normal"))

        self.simulation_statistics_turtle.goto(X_POSITION, Y_POSITION - 25)
        self.simulation_statistics_turtle.write(f"O-RUs: {self.num_rus*self.num_dus}", align="left", font=("Arial", 16, "normal"))

        self.simulation_statistics_turtle.goto(X_POSITION, Y_POSITION - 50)
        self.simulation_statistics_turtle.write(f"O-DUs: {self.num_dus}", align="left", font=("Arial", 16, "normal"))

        self.simulation_statistics_turtle.goto(X_POSITION, Y_POSITION - 75)
        self.simulation_statistics_turtle.write(f"UEs: {self.num_ues}", align="left", font=("Arial", 16, "normal"))

        self.simulation_statistics_turtle.goto(X_POSITION, Y_POSITION - 100)
        self.simulation_statistics_turtle.write(f"Energy Consumption: {round(self.get_total_energy_consumption()/1e+3,4)} kWh", align="left", font=("Arial", 16, "normal"))

    def update_component_connection_display(self):
        
        ru: O_RU
        for ru in self.rus.values():
            self.ru_du_connection_turtle.penup()
            if ru.get_du():
                self.ru_du_connection_turtle.goto(ru.get_du().get_position().x/GRAPHICAL_SCALING_FACTOR, ru.get_du().get_position().y/GRAPHICAL_SCALING_FACTOR)
                self.ru_du_connection_turtle.pendown()
                self.ru_du_connection_turtle.goto(ru.get_position().x/GRAPHICAL_SCALING_FACTOR, ru.get_position().y/GRAPHICAL_SCALING_FACTOR)

        for ue in self.ues.values() :
            if ue.get_ru():
                self.ue_connection_turtle.goto(ue.get_position().x/GRAPHICAL_SCALING_FACTOR, ue.get_position().y/GRAPHICAL_SCALING_FACTOR)
                self.ue_connection_turtle.pendown()
                self.ue_connection_turtle.goto(ue.get_ru().get_position().x/GRAPHICAL_SCALING_FACTOR, ue.get_ru().get_position().y/GRAPHICAL_SCALING_FACTOR)
                self.ue_connection_turtle.penup()
                
    def calculate_average_fronthaul_delay(self) -> float:
        total_delay = 0.0
        count = 0
        
        ru: O_RU
        for ru in self.rus.values():
            if ru.get_du():
                delay = calculate_fronthaul_delay(ru, ru.get_du())
                total_delay += delay
                count += 1
        return total_delay / count if count > 0 else 0.0

    def initialize_components(self):
        self.rus = {}
        self.dus = {}
        self.ues = {}
        
        np.random.seed(self.seed)
        rng = np.random.default_rng(seed=self.seed)
        for du in range(self.num_dus):
            # Create m DUs, assign IDs to them
            # Place the DUs automatically
            D_THETA = np.rad2deg(np.pi*du/self.num_dus)
            newDU = O_DU(Point(DU_DISTANCE_FROM_CENTER*np.cos(D_THETA),DU_DISTANCE_FROM_CENTER*np.sin(D_THETA)))
            self.dus[du] = newDU
            
        for ru in range(self.num_rus):
            R_THETA = np.rad2deg(2*np.pi*ru/self.num_rus)
            newRU = O_RU(create_random_point(self.simulation_side_length/2))
            self.rus[len(self.rus)] = newRU
            newRU.connect_du(self.dus[np.random.randint(0,self.num_dus)]) # Randomly connect RU to DU

        for id in range(self.num_ues):
            # Create k UEs, assign IDs to them.
            newUE = UE(create_random_point(self.simulation_side_length/2))
            self.ues[id] = newUE

        if self.graphics:
            self.screen: turtle._Screen = turtle.Screen()
            self.screen.setup(width=1500,height=1500)
            self.screen.title("Stochastic DQN Model for Joint Optimization of Delay and Energy Efficiency Simulation")
            self.screen.tracer(0)

            self.screen.register_shape(UE_IMAGE)
            self.screen.register_shape(RU_IMAGE)
            self.screen.register_shape(DU_IMAGE)
            self.screen.register_shape(RU_OFF_IMAGE)
            self.screen.register_shape(DU_OFF_IMAGE)

            self.ue_connection_turtle = turtle.Turtle()
            self.ue_connection_turtle.speed(0)
            self.ue_connection_turtle.penup()
            self.ue_connection_turtle.pencolor("blue")
            self.ue_connection_turtle.hideturtle()

            self.ru_du_connection_turtle = turtle.Turtle()
            self.ru_du_connection_turtle.speed(0)
            self.ru_du_connection_turtle.penup()
            self.ru_du_connection_turtle.pencolor("green")
            self.ru_du_connection_turtle.hideturtle()

            self.simulation_statistics_turtle = turtle.Turtle()
            self.simulation_statistics_turtle.speed(0)
            self.simulation_statistics_turtle.penup()
            self.simulation_statistics_turtle.hideturtle()

    def write_results(self)->None:
        data = {
            'Seed': [seed],
            'Simulation Length (seconds)': [self.simulation_length],
            'Total Energy Consumption (kWh)': round(self.get_total_energy_consumption()/1e+3,4),
            'RU Geolocations': [[(float(ru.get_position().x),float(ru.get_position().y)) for ru in self.rus.values()]],
            'DU Geolocations': [[(float(du.get_position().x),float(du.get_position().y)) for du in self.dus.values()]]
            }
        df = pd.DataFrame(data)
        df.to_csv(f'../data/output{datetime.now()}')
        
    def update_ues(self)->None:
        
        ue: UE
        for ue in self.ues.values():
            ue.walk(create_random_point(8))
            best_connected_ru = ue.get_ru() if ue.get_ru() else self.rus[0]
            best_connected_ru_rsrp = rsrp(best_connected_ru, ue)

            unit: O_RU
            for unit in self.rus.values():
                if unit.active:
                    tentative_rsrp = rsrp(unit, ue)
                    #print(tentativeRSRP)
                    if tentative_rsrp >= RSRP_THRESHOLD_DBM:
                        best_connected_ru = unit
                        best_connected_ru_rsrp = rsrp(best_connected_ru, ue)

            if best_connected_ru_rsrp < RSRP_THRESHOLD_DBM:
                ue.detach_from_ru()
            else:
                ue.attach_to_ru(best_connected_ru)
        
    def generate_action_space(self)->torch.Tensor:
        ru_sleep_space = torch.tensor([1 if unit.awake() else 0 for unit in self.rus.values()])
        du_sleep_space = torch.tensor([1 if unit.awake() else 0 for unit in self.dus.values()])
        ru_du_swap_space = torch.tensor([1 if ru.getDU() == du else 0 for ru, du in product(self.rus.values(),self.dus.values())])
        return torch.cat((ru_sleep_space, du_sleep_space, ru_du_swap_space), 0)

    def step(self, step: int)->None:
        self.main_loop_step = step
        if self.graphics:
            self.ue_connection_turtle.clear()
            self.ru_du_connection_turtle.clear()
            self.simulation_statistics_turtle.clear()
            self.update_statistics_display(step)
            self.update_component_connection_display()

        self.update_ues()
        self.update_total_energy_consumption()
        if self.graphics:
            self.screen.update()
        
    def run(self, simulation_length: int)->None:
        self.initialize_components()
        self.simulation_length = simulation_length
        self.running = True
        
        # Main loop
        for _ in range(0, self.simulation_length):
            time.sleep(self.time_step_length)
            self.step(_)
        self.write_results()


# FUNCTIONS

def calculate_fronthaul_delay(ru: O_RU, du: O_DU)->float:
    # This function generates the fronthaul delay based on the paper "Dynamic Placement of O-CU and O-DU Functionalities in Open-RAN Architecture" by Hojeij et al.
    # The delay is calculated as the distance between RU and DU divided by the speed of light, plus a processing load factor.
    return ru.get_position().dist(du.get_position())/c + du.get_processing_load()*0.035 + 0.4 * rng.uniform(0.025,0.25)

def path_loss(ru: O_RU, ue: UE)->float:
    #UMi path loss based on https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.00.00_60/tr_138901v180000p.pdf
    
    hprime_bs = ru.height - 1.5
    
    d_2d = ru.get_position().dist(ue.get_position())
    d_3d = np.sqrt(pow((ru.get_position().x-ue.get_position().x),2) + pow((ru.get_position().y-ue.get_position().y),2) + pow((ru.height),2))
    d_bp = 4*hprime_bs*1.5*ru.operating_frequency/c


    pl1 = 32.4 + 21*np.log10(d_3d)+20*np.log10(ru.operating_frequency)
    pl2 = 32.4 + 40*np.log10(d_3d)+20*np.log10(ru.operating_frequency)-9.5*np.log10(d_bp**2+(ru.height-1.5)**2)

    return pl1 if d_2d > 10 and d_2d < d_bp else pl2

def rsrp(ru: O_RU, ue: UE)->float:
    # Calculation based on "Realistic Signal Strength Simulation for ORAN Testing Environments" by Nour Bahtite
    return ru.transmission_power - path_loss(ru, ue)

def create_random_point(s)->Point:
    return Point(rng.uniform(-1*s,s),rng.uniform(-1*s,s))