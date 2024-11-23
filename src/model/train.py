import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


import numpy as np
from src.environment.FactoryModel import factory_model
from src.model.dqn_agent import DQNAgent

from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

# Define global flags for verbose and visualization modes
VERBOSE = False  # Manually set to True or False to control verbose output
VISUALIZATION = True  # Manually set to True or False to enable visualization

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


def start_visualization():
    grid = CanvasGrid(agent_portrayal, 20, 10, 500, 250)
    server = ModularServer(
        factory_model,
        [grid],
        "Factory Infection Model",
        {"width": 20, "height": 10, "N": 50, "visualization": True}  # Enable visualization mode
    )
    server.port = 8521
    server.launch()

def run_training():
    """Runs the training loop with optional verbose output for each episode."""
    if VERBOSE:
        print("Verbose mode is on")

    # Initialize parameters
    state_dim = 4  # Example state dimensions: [healthy, infected, recovered, vaccinated]
    action_dim = 3  # Action space: mask mandate, social distancing, vaccination
    num_episodes = 500
    target_update_freq = 10

    # Initialize DQN agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    # Training loop
    for episode in range(num_episodes):
        env = factory_model(width=20, height=10, N=50)
        state = env.get_state()
        total_reward = 0

        # Run an episode with a maximum of 100 steps
        for t in range(100):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Print progress if verbose is enabled
        if VERBOSE:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    print("Training complete.")


# Optional Command-line Argument Parsing
try:
    import argparse
    parser = argparse.ArgumentParser(description="Run Factory Infection Model with options.")
    parser.add_argument("--visualize", action="store_true", help="Run with interactive visualization.")
    parser.add_argument("--verbose", action="store_true", help="Print episode progress during training.")
    args = parser.parse_args()

    # Override global flags with command-line arguments if provided
    VERBOSE = args.verbose if args.verbose else VERBOSE
    VISUALIZATION = args.visualize if args.visualize else VISUALIZATION

except ImportError:
    print("argparse module not found, using defaults for VERBOSE and VISUALIZATION")


# Run based on provided or manually set flags
if VISUALIZATION:
    start_visualization()
else:
    run_training()
