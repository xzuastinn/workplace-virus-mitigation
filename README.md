# Simulating the Spread of COVID-19 in a Largely Populated Workforce

`FactoryModel.py` Class that implements a comprehensive simulation of a factory environment. Simulation looks at the intersection of worker health, productivity, and health policy measures a factory owner can implement. Key parameters that we look at are: mask mandates, social distancing mandate, testing levels, cleaning protocols, and plexiglass dividers to split the factory floor into sections. 

`FactoryConfig.py` Configuration manager for the factory simulation providing flexibility to tweak the model parameters listed above. Has static configurations for the server visualization and running a RL model of the environment. Allows dynamic updates to the simulation's configurations allowing the RL model to update health protocols for the factory. 

`GridManager.py` Manages the grid environment of the factory simulation. Organizes the workplace into configurable sections, determines agent placement, and manages section infection level and cleaning processes. Cleaning levels reduce section infection level at the cost of some productivity. 

`Quarantine.py` Handles the quarantine process of agents in the simulation model. Stops the spread of an infected sick agent by putting them into quarantine where they will stay until they reach the recovered state. During time in quarantine, agents will have 0 production output. 

`Run.py` runs the model with visualization. 

`Server.py` Class to run the simulation in a visual display with graphs that track the spread of the virus in real time and the result it has on the factory's production output.

`Testing.py` Manages health testing procedures by using the current testing configuration to test a certain proportion of agents within the environment. Each testing level has its own impact to productivity and how many tests get conducted and when. Integrated with Quarantine manager to send positive tested agents into quarantine. 

`train.py` runs the DQN model; toggle for visualization and verbosity 

`WorkerAgent.py` Class that handles all agent construction and activities during the simulation. Agents can be healthy, infected, recovered, or can face death. They have their own unique base productivity level that gets impacted based on health protocols implemented by the FactoryConfig.py class. Agents are assigned sections within the grid and are confined to a 2by2 workspace for each shift. 
Infection probability to spread is calculated within this class on an agent level depending on the Manhattan distance an infected agent is from a healthy agent. 

In each simulation step, the agent performs the following:

Moves to a valid position within its section.
Spreads infection to nearby agents based on proximity and environmental factors.
Updates infection status based on time spent in each health state.
Recalculates production output based on current conditions.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Features


## Installation

### Prerequisites

- Python 3.10 or higher
- [miniconda3](https://docs.anaconda.com/miniconda/miniconda-install/)
- VSCode, PyCharm, or your preferred IDE

### Clone the Repository

If you are using GitHub Personal Access Token, follow the tutorial [here](https://kettan007.medium.com/how-to-clone-a-git-repository-using-personal-access-token-a-step-by-step-guide-ab7b54d4ef83)
```bash
git clone https://github.khoury.northeastern.edu/samp3209/CS5100Final.git
cd CS5100Final
```

### Set up Python Environment

1. Create and activate a new Conda environment:
```bash
conda create --name virus_sim python
conda activate virus_sim
```

If you've already created the environment, simply activate it:
```bash
conda activate virus_sim
```

2. Install required packages:
```bash 
pip install -e .
```
OR 

```bash
pip install -r requirements.txt
```
Would suggest using the first option as it is easier to manage dependencies.

Note: If you encounter any issues with the installation, make sure your conda and pip are up to date:
```bash
conda update conda
pip install --upgrade pip
```

## Usage



