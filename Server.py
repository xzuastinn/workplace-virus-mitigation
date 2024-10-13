from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
#from mesa.visualization.UserParam import UserSettableParameter
from WorkerModel import WorkerModel

"""Still need to setup https://mesa.readthedocs.io/stable/tutorials/visualization_tutorial.html"""

def agent_portrayal(agent):
    portrayal = {"Shape": "Circle", "Filled": "true", "r": 0.5}

    if agent.health == 1:
        portrayal["Color"] = "green"
        portrayal["Layer"] = 0
    return portrayal

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
server = ModularServer(WorkerModel,
                       [grid],
                       "Worker Model",
                       {"N":10, "width":10, "height":10})
server.port = 8524
server.launch()