import random
from mesa import Agent

class WorkerAgent(Agent):
    """A worker agent starting in healthy state. Has an infection chance associated with it 
    and a prefered side of the factor to work on."""
    def __init__(self, unique_id, model, side):
        super().__init__(unique_id, model)
        self.side = side
        self.health_status = "healthy"
        self.infenction_chance = 0.1
        self.infection_time = 0
    
    def move(self):
        """Function to generate possible moves then take a random action."""
        x, y = self.pos

        possible_moves = [
            pos for pos in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            if (self.side == 'left' and pos[0] < self.model.central_line) or
            (self.side == 'right' and pos[0] >= self.model.central_line)
        ]

        if possible_moves:
            new_move = self.random.choice(possible_moves)
            self.model.grid.move_agent(self, new_move)

    def infection(self): 
        """Infection spread based on probability
            Should increase probability based on time and proximity spent near and infected agent
        """
        if self.health_status == 'infected':
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for other in cellmates:
                if isinstance(other, WorkerAgent):
                    if other.health_status == 'healthy':
                        if self.random.random() <= 0.2: # 20% chance to infect
                            other.health_status = 'infected'
    
    def update_health(self):
        """Updates the health of a sick worker. 
            Could potentially add in a 4th state of deceased worker
            Should implement some kind of probability associated with 
            healing faster/slower
            """
        if self.health_status == 'infected':
            self.infection_time += 1
            if self.infection_time > 14:
                self.health_status = 'recovered'
                

    def step(self) -> None:
        self.move()
        self.infection()
        self.update_health()

    def give_health(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.health += 1
            self.health -= 1

def EnvironmentVisualization(agent):
    """Sets the properties(qualities) for each agent to show up with in the gui
    Also controls the visualization of the environment."""
    if agent is None: # Doesn't work currently
        center = agent.model.grid.width // 2 if agent and agent.model else 0
        x = getattr(agent, 'pos', (0, 0))[0] if agent else 0
        if x == center:
            return {
                "Shape": "rect",
                "Color": "black",
                "Filled": True,
                "Layer": 1,
                "w": 0.55,  #size of the line
                "h": 1
            }
        return None
    
    qualities = {"Shape": "circle", "Filled": "true", "Layer": 0, "r":0.5}
    
    if agent.health_status == "healthy": 
        qualities["Color"] = "green"
        qualities["Layer"] = 1
    elif agent.health_status == "infected":
        qualities["Color"] = "red"
        qualities["Layer"] = 2
    elif agent.health_status == "recovered":
        qualities["Color"] = "blue"
        qualities["Layer"] = 3

    return qualities

class HomeAgent(Agent):
    """Agents that exist solely in a workers home"""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.health_status = "healthy"
    
    def step(self):
        pass
