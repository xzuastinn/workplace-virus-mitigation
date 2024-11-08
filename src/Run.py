from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from environment.FactoryModel import FactoryModel


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


# Set up the grid size and visualization
grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)


# Create a server for the model
server = ModularServer(
    FactoryModel,
    [grid],
    "Factory Infection Model",
    {"width": 10, "height": 10, "N": 10}  # Model parameters: grid size and number of agents
)

# Launch the server to visualize the environment
server.launch()
