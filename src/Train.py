import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
from mesa.visualization.modules import CanvasGrid, ChartModule
from environment.FactoryModel import factory_model
from src.model.dqn_agent import DQNAgent
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from environment.FactoryModel import factory_model
from environment.FactoryConfig import FactoryConfig

def agent_portrayal(agent):
    """Defines how agents appear in the visualization."""
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    if hasattr(agent, "health_status"):
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

# Visualization components
GRID_WIDTH = 50
GRID_HEIGHT = 25
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 250
grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT)


# Create the server for visualization
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
    visualization=False
)


chart = ChartModule(
    [
        {"Label": "Healthy", "Color": "Green"},
        {"Label": "Infected", "Color": "Red"},
        {"Label": "Recovered", "Color": "Blue"},
        {"Label": "Death", "Color": "Black"}
    ]
)
prod_chart = ChartModule([{"Label": "Productivity", "Color": "Purple"}])
daily_infections_chart = ChartModule([{"Label": "Daily Infections", "Color": "Red"}])

#PARAMETERS
cleaning_options = ['light', 'medium', 'heavy']
splitting_options = [0, 1, 2, 3]  # none, half, quarter, eighth
testing_options = ['none', 'light', 'medium', 'heavy']
social_distancing_options = [False, True]
mask_mandate_options = [False, True]
shifts_options = [1, 2, 3, 4]  # maps to 1, 2, 3, or 4 shifts per day

# Generate all combinations
combinations = list(itertools.product(
    cleaning_options,
    splitting_options,
    testing_options,
    social_distancing_options,
    mask_mandate_options,
    shifts_options
))

actions = [
    {
        'cleaning_type': cleaning,
        'splitting_level': splitting,
        'testing_level': testing,
        'social_distancing': social_distancing,
        'mask_mandate': mask_mandate,
        'shifts_per_day': shifts
    }
    for cleaning, splitting, testing, social_distancing, mask_mandate, shifts in combinations
]

state_dim = 8
action_dim = len(actions)
agent = DQNAgent(state_dim, action_dim)

#TRAINING PARAMETERS 
num_episodes = 2000
max_steps_per_episode = 240 #10 Days

def find_empty_cell(model, x_start=None, x_end=None):
    """HELPER FUNCTION. Find an empty cell in the grid within the given x bounds"""
    width = model.grid.width
    height = model.grid.height
    
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = width

    empty_cells = []
    for x in range(x_start, x_end):
        for y in range(height):
            if model.grid.is_cell_empty((x, y)):
                empty_cells.append((x, y))
    
    if empty_cells:
        return random.choice(empty_cells)
    return None

def train_with_toggle(dqn_agent, num_episodes, max_steps_per_episode, visualize_every=50, enable_visualization=False):
    """MAIN TRAINING LOOP"""
    total_cleaning_counter = {"light": 0, "medium": 0, "heavy": 0}
    total_shifts_counter = {"1": 0, "2": 0, "3": 0, "4": 0}
    total_mask_counter = {True: 0, False: 0}
    total_splitting_level_counter = {"0": 0, "1": 0, "2": 0, "3": 0}
    total_swab_testing_counter = {"none": 0, "light": 0, "medium": 0, "heavy": 0}
    total_social_distancing_counter = {True: 0, False: 0}

    for episode in range(num_episodes):
        is_visualizing = enable_visualization and (episode % visualize_every == 0)
        model = factory_model(
            width=GRID_WIDTH,
            height=GRID_HEIGHT,
            N=100,
            config=viz_config if is_visualizing else None,
            visualization=is_visualizing
        )

        if is_visualizing:
            print(f"Starting visualization for episode {episode + 1}")
            server = ModularServer(
                factory_model,
                [grid, chart, prod_chart, daily_infections_chart],
                "Factory Infection Model",
                {"N": 100, "config": viz_config, "width": GRID_WIDTH, "height": GRID_HEIGHT},
            )
            server.port = 8511
            server.launch()

        state = np.array(model.get_state())
        total_reward = 0

        for step in range(max_steps_per_episode):
            if step % 24 == 0:
                action_index = dqn_agent.select_action(state)
                action = actions[action_index]
                #PROBABLY SHOULD BE MOVED TO GRIDMANAGER. HANDLES NEW SPLIT CHANGE BORDERS
                if 'splitting_level' in action:
                    old_level = model.grid_manager.splitting_level
                    if old_level != action['splitting_level']:
                        positions = model.grid_manager.get_random_positions(model.num_agents)
                        active_agents = [agent for agent in model.schedule.agents 
                                       if not agent.is_dead and not agent.is_quarantined]
                        
                        #remove all active agents
                        for agent in active_agents:
                            if agent.pos is not None:
                                model.grid.remove_agent(agent)
                                agent.pos = None
                        
                        #Place active agents in their new positions
                        for i, agent in enumerate(active_agents):
                            if i < len(positions):
                                new_pos = positions[i]
                                if model.grid.is_cell_empty(new_pos):
                                    model.grid.place_agent(agent, new_pos)
                                    agent.set_base_position(new_pos)
                                else:
                                    x_start = (model.grid.width // (2 ** action['splitting_level'])) * (i % (2 ** action['splitting_level']))
                                    x_end = x_start + (model.grid.width // (2 ** action['splitting_level']))
                                    empty_pos = find_empty_cell(model, x_start, x_end)
                                    if empty_pos:
                                        model.grid.place_agent(agent, empty_pos)
                                        agent.set_base_position(empty_pos)
                
                model.update_config(action)

                for agent in model.schedule.agents:
                    if not agent.is_dead and not agent.is_quarantined and agent.pos is None:
                        empty_pos = find_empty_cell(model)
                        if empty_pos:
                            model.grid.place_agent(agent, empty_pos)
                            agent.set_base_position(empty_pos)

            step_results = model.step()

            infected = step_results.get('infected', 0)
            productivity = step_results.get('productivity', 0)
            death = step_results.get('death', 0)

            reward = (-20 * infected) - (100 * death)  #Reduced penalty multipliers
            if productivity >= 0.75:
                reward += 8 * productivity
            elif productivity >= 0.6:
                reward += 2 * productivity
            elif productivity < 0.6:
                reward -= 100 * (0.6 - productivity)

            total_reward += reward

            next_state = np.array(model.get_state())
            done = model.stats.is_done()
            if step % 24 == 0:
                dqn_agent.store_experience(state, action_index, reward, next_state, done)
                dqn_agent.train()
            state = next_state

            if done:
                break

        # Update the target network periodically
        if episode % 10 == 0:
            dqn_agent.update_target_network()
        dqn_agent.rewards_history.append(total_reward)

        # Print progress
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {dqn_agent.epsilon:.4f}")
        # COUNTER PRINTS AT EVERY EPISODE
        print(f"  Cleaning Counter: {model.cleaning_counter}")
        print(f"  Shifts Counter: {model.shifts_counter}")
        print(f"  Mask Counter: {model.mask_counter}")
        print(f"  Splitting Level Counter: {model.splitting_level_counter}")
        print(f"  Swab Testing Counter: {model.swab_testing_counter}")
        print(f"  Social Distancing Counter: {model.social_distancing_counter}")
        
        # Update total counters
        for key in total_cleaning_counter:
            total_cleaning_counter[key] += model.cleaning_counter[key]
        for key in total_shifts_counter:
            total_shifts_counter[key] += model.shifts_counter[key]
        for key in total_mask_counter:
            total_mask_counter[key] += model.mask_counter[key]
        for key in total_splitting_level_counter:
            total_splitting_level_counter[key] += model.splitting_level_counter[key]
        for key in total_swab_testing_counter:
            total_swab_testing_counter[key] += model.swab_testing_counter[key]
        for key in total_social_distancing_counter:
            total_social_distancing_counter[key] += model.social_distancing_counter[key]

    # Save the trained model
    dqn_agent.save_model("dqn_factory_model.pth")
    #PRINTS FOR TOTAL COUNTS AFTER TRAINING FINISHED
    print("Training completed. Model saved as 'dqn_factory_model.pth'.")
    print(f"\nTotal Cleaning Counter: {total_cleaning_counter}")
    print(f"Total Shifts Counter: {total_shifts_counter}")
    print(f"Total Mask Counter: {total_mask_counter}")
    print(f"Total Splitting Level Counter: {total_splitting_level_counter}")
    print(f"Total Swab Testing Counter: {total_swab_testing_counter}")
    print(f"Total Social Distancing Counter: {total_social_distancing_counter}")
    #FINAL PLOTS
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(dqn_agent.q_values_history)
    plt.title('Final Average Q-Values')
    plt.xlabel('Training Steps')
    plt.ylabel('Q-Value')
    
    plt.subplot(2, 2, 2)
    plt.plot(dqn_agent.losses_history)
    plt.title('Final Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(dqn_agent.epsilons_history)
    plt.title('Final Epsilon Decay')
    plt.xlabel('Training Steps')
    plt.ylabel('Epsilon')
    
    plt.subplot(2, 2, 4)
    plt.plot(dqn_agent.rewards_history)
    plt.title('Final Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig('final_training_metrics.png')
    plt.close()

train_with_toggle(agent, num_episodes, max_steps_per_episode, visualize_every=5, enable_visualization=False)