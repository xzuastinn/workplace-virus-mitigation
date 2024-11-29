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

def factory_model_with_dqn(N, config, width, height, dqn_agent, action_space):
    """Creates and returns a factory model that uses the trained DQN for decision making"""
    model = factory_model(
        width=width,
        height=height,
        N=N,
        config=config,
        visualization=True
    )
    
    # Store the original step function
    original_step = model.step
    last_day = -1  # Track the last day we made a decision
    
    def new_step():
        nonlocal last_day
        current_day = model.current_day
        
        # Make decisions at the start of each new day
        if current_day > last_day:
            last_day = current_day
            state = np.array(model.get_state())
            state_tensor = torch.FloatTensor(state)
            
            with torch.no_grad():
                # Force some exploration during visualization
                action_index = dqn_agent.select_action(state_tensor, train=False)
                action = action_space[action_index]
            
            print(f"\nDay {current_day} - Current State:")
            print(f"Healthy: {state[0]}, Infected: {state[1]}, Recovered: {state[2]}, Dead: {state[3]}")
            print(f"Productivity: {state[4]:.2f}, Step in Day: {state[5]}")
            print(f"Social Distancing: {bool(state[6])}, Mask Mandate: {bool(state[7])}")
            
            print("\nCurrent Configuration:")
            print(f"Cleaning: {model.initial_cleaning}")
            print(f"Splitting: {model.grid_manager.splitting_level}")
            print(f"Testing: {model.test_lvl}")
            print(f"Social Distancing: {model.social_distancing}")
            print(f"Mask Mandate: {model.mask_mandate}")
            print(f"Shifts: {model.shifts_per_day}")
            
            print("\nProposed Configuration:")
            print(f"Cleaning: {action['cleaning_type']}")
            print(f"Splitting: {action['splitting_level']}")
            print(f"Testing: {action['testing_level']}")
            print(f"Social Distancing: {action['social_distancing']}")
            print(f"Mask Mandate: {action['mask_mandate']}")
            print(f"Shifts: {action['shifts_per_day']}")
            
            # Check if the configuration actually changed
            config_changed = (
                action['cleaning_type'] != model.initial_cleaning or
                action['splitting_level'] != model.grid_manager.splitting_level or
                action['testing_level'] != model.test_lvl or
                action['social_distancing'] != model.social_distancing or
                action['mask_mandate'] != model.mask_mandate or
                action['shifts_per_day'] != model.shifts_per_day
            )
            
            if config_changed:
                print("\nApplying configuration changes...")
                # Apply the grid splitting changes first if needed
                if action['splitting_level'] != model.grid_manager.splitting_level:
                    active_agents = [agent for agent in model.schedule.agents 
                                   if not agent.is_dead and not agent.is_quarantined]
                    
                    # Remove all active agents
                    for agent in active_agents:
                        if agent.pos is not None:
                            model.grid.remove_agent(agent)
                            agent.pos = None
                    
                    # Update splitting level and get new positions
                    model.splitting_level = action['splitting_level']
                    positions = model.grid_manager.get_random_positions(len(active_agents))
                    
                    # Place agents in new positions
                    for i, agent in enumerate(active_agents):
                        if i < len(positions):
                            new_pos = positions[i]
                            if model.grid.is_cell_empty(new_pos):
                                model.grid.place_agent(agent, new_pos)
                                agent.set_base_position(new_pos)
                
                model.update_config(action)
            else:
                print("\nNo configuration changes needed")
        
        # Call the original step function
        return original_step()
    
    # Replace the model's step function
    model.step = new_step
    
    return model


# Visualization Server
server = ModularServer(
    factory_model_with_dqn,
    [grid, chart, prod_chart, daily_infections_chart],
    "Factory Infection Model with DQN",
    {
        "N": 100, 
        "config": viz_config, 
        "width": GRID_WIDTH, 
        "height": GRID_HEIGHT,
        "dqn_agent": agent,  # Pass the DQN agent
        "action_space": actions  # Pass the action space
    }
)

server.port = 8511
server.launch()
