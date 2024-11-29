from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from environment.FactoryModel import factory_model
from environment.FactoryConfig import FactoryConfig
from src.model.dqn_agent import DQNAgent
import torch
import numpy as np
import itertools

cleaning_options = ['light', 'medium', 'heavy']  # light, medium, heavy
splitting_options = [0, 1, 2, 3]  # none, half, quarter, eighth
testing_options = ['none', 'light', 'medium', 'heavy']  # none, light, medium, heavy
social_distancing_options = [False, True]
mask_mandate_options = [False, True]
shifts_options = [1, 2, 3, 4]  # maps to 1, 2, 3, or 4 shifts per day

# Generate all combinations of actions
actions = [
    {
        'cleaning_type': cleaning,
        'splitting_level': splitting,
        'testing_level': testing,
        'social_distancing': social_distancing,
        'mask_mandate': mask_mandate,
        'shifts_per_day': shifts
    }
    for cleaning, splitting, testing, social_distancing, mask_mandate, shifts in itertools.product(
        cleaning_options,
        splitting_options,
        testing_options,
        social_distancing_options,
        mask_mandate_options,
        shifts_options
    )
]


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

daily_infections_chart = ChartModule([
    {"Label": "Daily Infections", "Color": "Red"}
], data_collector_name='datacollector')

# Factory configuration for visualization
viz_config = FactoryConfig(
    width=GRID_WIDTH, 
    height=GRID_HEIGHT, 
    num_agents=100,
    splitting_level=0,
    cleaning_type='light',
    testing_level='light',
    social_distancing=False,
    mask_mandate=False,
    shifts_per_day=4,
    steps_per_day=24,
    visualization=True
)

# Load the trained DQN model
state_dim = 8  # Ensure this matches your state dimension
action_dim = len(actions)  # Number of possible actions
agent = DQNAgent(state_dim, action_dim)
agent.load_model("dqn_factory_model.pth")  # Load the trained weights

# Create a server for the model
def factory_model_with_dqn(N, config, width, height):
    model = factory_model(
        width=width,
        height=height,
        N=N,
        config=config,
        visualization=True
    )
    
    state = np.array(model.get_state())
    for step in range(model.steps_per_day):  # Run for one day
        # Select action using the trained model
        action_index = agent.select_action(state, train=False)  # Use train=False to disable exploration
        action = actions[action_index]

        # Apply action and advance the simulation
        model.update_config(action)
        step_results = model.step()
        state = np.array(model.get_state())
    return model

# Visualization Server
server = ModularServer(
    factory_model_with_dqn,
    [grid, chart, prod_chart, daily_infections_chart],
    "Factory Infection Model with DQN",
    {"N": 100, "config": viz_config, "width": GRID_WIDTH, "height": GRID_HEIGHT}
)

server.port = 8511
server.launch()
