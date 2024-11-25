from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from environment.FactoryModel import factory_model



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
    elif agent.health_status == "death":
        portrayal["Color"] = "black"
        portrayal["Layer"] = 4

    return portrayal

GRID_WIDTH = 50
GRID_HEIGHT = 25
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 250
# Set up the grid size and visualization
grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT)

# Create the chart 
chart = ChartModule( 
    [
        {"Label": "Healthy", "Color": "Green"},
        {"Label": "Infected", "Color": "Red"},
        {"Label": "Recovered", "Color": "Blue"},
        {"Label": "Death", "Color": "Black"}
    ]
)

prod_chart = ChartModule([
    {"Label": "Productivity", "Color": "Purple"},
], data_collector_name='datacollector')


daily_infections_chart = ChartModule([ #creates STEP infections chart not daily
    {"Label": "Daily Infections", "Color": "Red"}
], data_collector_name='datacollector')

# viz_config = FactoryConfig.get_default_viz_config()

# Create a server for the model
server = ModularServer(
    factory_model,
    [grid, chart, prod_chart, daily_infections_chart],
    "Factory Infection Model",
    {"width": GRID_WIDTH, 
     "height": GRID_HEIGHT, 
     "N": 100, 
     "visualization": True,
     }
)

server.port = 8511
# Launch the server to visualize the environment
server.launch()
