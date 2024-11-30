from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from environment.FactoryModel import factory_model
from environment.FactoryConfig import FactoryConfig

def agent_portrayal(agent):
    """Defines how agents appear in the visualization."""
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.7}

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

GRID_WIDTH = 10
GRID_HEIGHT = 10
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT)

health_chart = ChartModule(
    [
        {"Label": "Healthy", "Color": "Green"},
        {"Label": "Infected", "Color": "Red"},
        {"Label": "Recovered", "Color": "Blue"}
    ]
)

productivity_chart = ChartModule([
    {"Label": "Productivity", "Color": "Purple"},
], data_collector_name='datacollector')

demo_config = FactoryConfig(
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    num_agents=10,
    splitting_level=0,
    cleaning_type='heavy',
    testing_level='medium',
    social_distancing=True,
    mask_mandate=False,
    shifts_per_day=4,
    steps_per_day=24,
    visualization=True
)

def create_factory_model(N, config, width, height):
    """Create the factory model with the specified configuration."""
    return factory_model(
        width=width,
        height=height,
        N=N,
        config=config,
        visualization=True
    )

server = ModularServer(
    create_factory_model,
    [grid, health_chart, productivity_chart],
    "Factory Infection Model Demo",
    {"N": 8, "config": demo_config, "width": GRID_WIDTH, "height": GRID_HEIGHT}
)

server.port = 8512
server.launch()