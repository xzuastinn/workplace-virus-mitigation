from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from environment.FactoryModel import factory_model
from environment.FactoryConfig import FactoryConfig
from src.model.dqn_agent import DQNAgent
import torch
import numpy as np
import itertools

cleaning_options = ['light', 'medium', 'heavy']
splitting_options = [0, 1, 2, 3]  # none, half, quarter, eighth
testing_options = ['none', 'light', 'medium', 'heavy']
social_distancing_options = [False, True]
mask_mandate_options = [False, True]
shifts_options = [1, 2, 3, 4]  # maps to 1, 2, 3, or 4 shifts per day

class CurrentConfig(TextElement):
    """Display current configuration as text"""
    
    def render(self, model):
        return f"""
        <b>Current Configuration:</b><br>
        Cleaning Type: {model.initial_cleaning}<br>
        Splitting Level: {model.splitting_level}<br>
        Testing Level: {model.test_lvl}<br>
        Social Distancing: {model.social_distancing}<br>
        Mask Mandate: {model.mask_mandate}<br>
        Shifts Per Day: {model.shifts_per_day}
        """
    
#ACTION DICTIONARY
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
config_chart = ChartModule([
    {"Label": "Cleaning Level", "Color": "Brown"},
    {"Label": "Splitting Level", "Color": "Purple"},
    {"Label": "Testing Level", "Color": "Orange"},
    {"Label": "Social Distancing", "Color": "Green"},
    {"Label": "Mask Mandate", "Color": "Blue"}
], data_collector_name='datacollector')

def agent_portrayal(agent):
    """Defines how agents appear in the visualization."""
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if agent.is_quarantined:
        portrayal["Color"] = "brown"
        portrayal["Layer"] = 5  
    elif agent.health_status == "healthy":
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

grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT)
current_config = CurrentConfig()

chart = ChartModule(
    [
        {"Label": "Healthy", "Color": "Green"},
        {"Label": "Infected", "Color": "Red"},
        {"Label": "Recovered", "Color": "Blue"},
        {"Label": "Death", "Color": "Black"},
        {"Label": "Quarantined", "Color": "Brown"}
    ]
)

prod_chart = ChartModule([
    {"Label": "Productivity", "Color": "Purple"},
], data_collector_name='datacollector')

daily_infections_chart = ChartModule([
    {"Label": "Daily Infections", "Color": "Red"}
], data_collector_name='datacollector')

#NOT USED 
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


state_dim = 8
action_dim = len(actions)
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
    
    original_step = model.step
    last_day = -1
    
    def new_step():
        nonlocal last_day
        current_day = model.current_day
        
        if current_day > last_day:
            last_day = current_day
            state = np.array(model.get_state())
            state_tensor = torch.FloatTensor(state)
            
            with torch.no_grad():
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
                if action['splitting_level'] != model.grid_manager.splitting_level:
                    active_agents = [agent for agent in model.schedule.agents 
                                   if not agent.is_dead and not agent.is_quarantined]
                    
                    for agent in active_agents:
                        if agent.pos is not None:
                            model.grid.remove_agent(agent)
                            agent.pos = None
                    
                    model.splitting_level = action['splitting_level']
                    positions = model.grid_manager.get_random_positions(len(active_agents))
                    
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
    [grid, current_config, chart, prod_chart, daily_infections_chart],
    "Factory Infection Model with DQN",
    {
        "N": 100, 
        "config": viz_config, 
        "width": GRID_WIDTH, 
        "height": GRID_HEIGHT,
        "dqn_agent": agent,
        "action_space": actions
    }
)

server.port = 8511
server.launch()
