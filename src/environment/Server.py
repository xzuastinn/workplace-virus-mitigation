from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from FactoryModel import factory_model

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

grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT)

# Create the chart 
chart = ChartModule ([
        {"Label": "Healthy", "Color": "Green"},
        {"Label": "Infected", "Color": "Red"},
        {"Label": "Recovered", "Color": "Blue"}
    ]
)

prod_chart = ChartModule([    
    {"Label": "Productivity", "Color": "Purple"},
])

daily_infections_chart = ChartModule([
    {"Label": "Daily Infections", "Color": "Red"}
], data_collector_name='datacollector')
server = ModularServer(
    factory_model,
    [grid, chart, prod_chart, daily_infections_chart],
    "Factory Infection Model",
    {"width": GRID_WIDTH, "height": GRID_HEIGHT, "N": 40, "visualization": True}
)

server.port = 8511
server.launch()