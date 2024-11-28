import itertools
import numpy as np
import time
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

# Create the server for visualization
viz_config = FactoryConfig(
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    num_agents=100,
    splitting_level=0,
    cleaning_type='light',
    testing_level='light',
    social_distancing=False,
    mask_mandate=0,
    shifts_per_day=4,
    steps_per_day=24,
    visualization=True
)

# # Define the training loop with optional visualization
# def train_with_visualization(num_episodes=1000, visualize_every=50):
#     for episode in range(num_episodes):
#         is_visualizing = (episode % visualize_every == 0)
#         model = factory_model(
#             width=GRID_WIDTH,
#             height=GRID_HEIGHT,
#             N=100,
#             config=viz_config if is_visualizing else None,
#             visualization=is_visualizing
#         )
#         if is_visualizing:
#             print(f"Starting visualization for episode {episode + 1}")

#             # Launch the server for the current episode
#             server = ModularServer(
#                 factory_model,
#                 [grid, chart, prod_chart, daily_infections_chart],
#                 "Factory Infection Model",
#                 {"N": 100, "config": viz_config, "width": GRID_WIDTH, "height": GRID_HEIGHT},
#             )
#             server.port = 8511
#             server.launch()
        
#         # Continue training process here (omit for brevity)
#         # Train agent, collect rewards, update state...

# train_with_visualization()

# Visualization components
GRID_WIDTH = 50
GRID_HEIGHT = 25
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 250
grid = CanvasGrid(agent_portrayal, GRID_WIDTH, GRID_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT)
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

# Define the possible values for each parameter
cleaning_options = ["light", "medium", "heavy"]  # light, medium, heavy
splitting_options = [0, 1, 2, 3]  # none, half, quarter, eighth
testing_options = ["none", "light", "medium", "heavy"]  # none, light, medium, heavy
social_distancing_options = [False, True]
mask_mandate_options = [0, 1, 2, 3]
shifts_options = [0, 1, 2, 3]  # maps to 1, 2, 3, or 4 shifts per day

# Generate all combinations
combinations = list(itertools.product(
    cleaning_options,
    splitting_options,
    testing_options,
    social_distancing_options,
    mask_mandate_options,
    shifts_options
))

# Convert combinations to dictionaries
actions = [
    {
        'cleaning_type': cleaning,
        'splitting_level': splitting,
        'testing_level': testing,
        'social_distancing': social_distancing,
        'mask_mandate': mask_mandate,
        'shifts_per_day"': shifts
    }
    for cleaning, splitting, testing, social_distancing, mask_mandate, shifts in combinations
]

# Initialize the environment and agent
state_dim = 8  # Adjust based on your `get_state` implementation
action_dim = len(actions)
agent = DQNAgent(state_dim, action_dim)

# Function to render the grid manually
def render_grid(grid, model):
    portrayal_data = []
    for cell_content in model.grid.coord_iter():
        if len(cell_content) == 3:  # Ensure proper tuple unpacking
            content, x, y = cell_content
            for obj in content:
                if hasattr(obj, "health_status"):  # Check for valid agent attributes
                    portrayal = grid.portrayal_method(obj)
                    portrayal_data.append(portrayal)
                else:
                    print(f"Unexpected content type: {type(obj)}")
    return portrayal_data

num_episodes = 10
max_steps_per_episode = 200

# Training loop with visualization
for episode in range(num_episodes):
    model = factory_model(width=GRID_WIDTH, height=GRID_HEIGHT, N=100, visualization=True)
    state = np.array(model.get_state())
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Select an action
        action_index = agent.select_action(state)
        action = actions[action_index]

        # Apply the action
        model.update_config(action)

        # Advance the simulation
        step_results = model.step()
        model.datacollector.collect(model)

        # Extract reward and state
        infected = step_results.get('infected', 0)  # Default to 0 if key is missing
        productivity = step_results.get('productivity', 0)     # Default to 0 if key is missing
        death = step_results.get('death', 0)
        
        # Base reward formula
        total_reward -= 1000 * infected  # Focus on penalizing infections

        total_reward -= 100000 * death

        # Incentivize higher productivity
        if productivity >= 0.6:
            total_reward += 1000 * productivity  # Larger reward for productivity above the threshold

        # Harsh penalty for productivity below 60%
        if productivity < 0.6:
            total_reward -= 1000 * productivity  # Significantly harsher penalty

        # Train the agent
        next_state = np.array(model.get_state())
        done = model.stats.is_done()
        agent.store_experience(state, action_index, total_reward, next_state, done)
        agent.train()
        state = next_state

        # Update visualization
        grid_state = render_grid(grid, model)
        print(f"Grid State: {grid_state}")  # Debugging the rendered grid state

        # Add a small pause for visualization purposes
        time.sleep(0.1)

        if done:
            break

    # Update the target network periodically
    if episode % 10 == 0:
        agent.update_target_network()

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

# Save the trained model
agent.save_model("dqn_factory_model.pth")

def train_with_toggle(num_episodes, visualize_every=50, enable_visualization=True):
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
            # Launch the server for the current episode
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
            # Select an action
            action_index = agent.select_action(state)
            action = actions[action_index]

            # Apply the action
            model.update_config(action)

            # Advance the simulation
            step_results = model.step()

            # Extract reward and state
            infected = step_results.get('infected', 0)
            productivity = step_results.get('productivity', 0)
            death = step_results.get('death', 0)

            # Reward calculation
            reward = -2 * infected - 100000 * death
            if productivity >= 0.6:
                reward += 20 * productivity
            if productivity < 0.6:
                reward -= 100000 * (0.6 - productivity)

            total_reward += reward

            # Train the agent
            next_state = np.array(model.get_state())
            done = model.stats.is_done()
            agent.store_experience(state, action_index, reward, next_state, done)
            agent.train()
            state = next_state

            if done:
                break

        # Update the target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        # Print progress
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Save the trained model
    agent.save_model("dqn_factory_model.pth")
    print("Training completed. Model saved as 'dqn_factory_model.pth'.")


train_with_toggle(num_episodes, visualize_every=5, enable_visualization=False)
