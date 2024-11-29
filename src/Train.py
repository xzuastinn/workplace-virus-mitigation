import itertools
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
    visualization=True
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

# Define the possible values for each parameter
cleaning_options = ['light', 'medium', 'heavy']  # light, medium, heavy
splitting_options = [0, 1, 2, 3]  # none, half, quarter, eighth
testing_options = ['none', 'light', 'medium', 'heavy']  # none, light, medium, heavy
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

# Convert combinations to dictionaries
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

# Initialize the environment and agent
state_dim = 8  # Adjust based on your `get_state` implementation
action_dim = len(actions)
agent = DQNAgent(state_dim, action_dim)

num_episodes = 10
max_steps_per_episode = 200

def train_with_toggle(num_episodes, max_steps_per_episode, visualize_every=50, enable_visualization=True):
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
            if step % 24 == 0:
                action_index = agent.select_action(state)
                action = actions[action_index]
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
            if step % 24 == 0:
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

        print(f"  Cleaning Counter: {model.cleaning_counter}")
        print(f"  Shifts Counter: {model.shifts_counter}")
        print(f"  Mask Counter: {model.mask_counter}")
        print(f"  Splitting Level Counter: {model.splitting_level_counter}")
        print(f"  Swab Testing Counter: {model.swab_testing_counter}")
        print(f"  Social Distancing Counter: {model.social_distancing_counter}")
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
    agent.save_model("dqn_factory_model.pth")
    print("Training completed. Model saved as 'dqn_factory_model.pth'.")
    print(f"\nTotal Cleaning Counter: {total_cleaning_counter}")
    print(f"Total Shifts Counter: {total_shifts_counter}")
    print(f"Total Mask Counter: {total_mask_counter}")
    print(f"Total Splitting Level Counter: {total_splitting_level_counter}")
    print(f"Total Swab Testing Counter: {total_swab_testing_counter}")
    print(f"Total Social Distancing Counter: {total_social_distancing_counter}")



train_with_toggle(num_episodes, max_steps_per_episode, visualize_every=5, enable_visualization=False)