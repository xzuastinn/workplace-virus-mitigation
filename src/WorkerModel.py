import random
import seaborn as sns
import numpy as np
import pandas as pd
from mesa import Agent, DataCollector, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from Agent import WorkerAgent, EnvironmentVisualization
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
    
class WorkerModel(Model):
    """Model with n agents"""

    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            a = WorkerAgent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
    
    def step(self):
        self.schedule.step()

class FactoryModel(Model):
    """Model that places agents in a grid with random activations """
    def __init__(self, width, height, N):
        super().__init__()  
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        self.central_line = width // 2 #Splits the environment in half

        first_infections = random.randrange(N) #Just start with one agent for now

        for i in range(self.num_agents):
            side = 'left' if random.random() < 0.5 else 'right'
            worker = WorkerAgent(i, self, side)
            if i == first_infections:
                worker.health_status = "infected"
            self.schedule.add(worker)

            x = self.random.randrange(self.central_line) if side == 'left' else self.random.randrange(self.central_line, width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(worker, (x, y))
            print(f"Agent {i} placed at position ({x}, {y})")
        
        self.datacollector = DataCollector(
            {
                "Healthy": lambda m: self.count_health_status(m, "healthy"),
                "Infected": lambda m: self.count_health_status(m, "infected"),
                "Recovered": lambda m: self.count_health_status(m, "recovered"),
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    @staticmethod
    def count_health_status(model, status):
        count = 0
        for agent in model.schedule.agents:
            if agent.health_status == status:
                count += 1
        return count

#width, height, num_workers = 20, 10, 50
#factory = FactoryModel(width, height, num_workers)
#for i in range(100):
    #factory.step()

gridwidth = 50
gridheight = 25
canvaswidth = 500
canvasheight = 250
grid = CanvasGrid(EnvironmentVisualization, gridwidth, gridheight, canvaswidth, canvasheight)

chart = ChartModule(
    [
        {"Label": "Healthy", "Color": "green"},
        {"Label": "Infected", "Color": "red"},
        {"Label": "Recovered", "Color": "blue"}
    ]
)


#Server initialization for visualization and graphing
server = ModularServer(
    FactoryModel,
    [grid, chart],
    "Factory Infection Model",
    {"width": gridwidth, "height": gridheight, "N": 50}
)

server.port = 8522
server.launch()
