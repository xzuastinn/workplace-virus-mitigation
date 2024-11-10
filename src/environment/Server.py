from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from FactoryModel import factory_model

"""Still need to setup https://mesa.readthedocs.io/stable/tutorials/visualization_tutorial.html"""

def agent_portrayal(agent):
    """Defines how agents appear in the visualization."""
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if agent.health_status == "healthy":
        portrayal["Color"] = "green"
        portrayal["Layer"] = 1
    elif agent.health_status == "infected":
        portrayal["Color"] = "red"
        portrayal["Layer"] = 2
    elif agent.health_status == "recovered":
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 3

    return portrayal

GRID_WIDTH = 25
GRID_HEIGHT = 25
CANVAS_WIDTH = 350
CANVAS_HEIGHT = 250

# Set up the grid size and visualization
grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT)

# Create the chart 
chart = ChartModule ([
        {"Label": "Healthy", "Color": "Green"},
        {"Label": "Infected", "Color": "Red"},
        {"Label": "Recovered", "Color": "Blue"}
    ]
)
reward_chart = ChartModule([
    {"Label": "Total Reward", "Color": "Orange"}
])
prod_chart = ChartModule([    
    {"Label": "Productivity", "Color": "Purple"},
])
server = ModularServer(
    factory_model,
    [grid, chart, reward_chart, prod_chart],
    "Factory Infection Model",
    {"width": GRID_WIDTH, "height": GRID_HEIGHT, "N": 75, "visualization": True}  # Model parameters: grid size and number of agents
)
server.port = 8511
server.launch()