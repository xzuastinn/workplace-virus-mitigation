# Simulating the Spread of COVID-19 in a Large Population Environment like a Workplace

`Model.py` is the script we use to create our simulation by adding agents to the grid.

`Agent.py` is the script we use to create our agent class and give it properties and rules on how it can move.

`Run.py` runs the model with no visualization

`Server.py` is what runs the web simulation 

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
pip install -r requirements.txt
```

Note: If you encounter any issues with the installation, make sure your conda and pip are up to date:
```bash
conda update conda
pip install --upgrade pip
```

## Usage



