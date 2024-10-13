from mesa import Agent

class WorkerAgent(Agent):
    """A worker agent starting in susceptible state."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.health = 1
    
    def step(self) -> None:
        self.move()
        if self.health > 0:
            self.give_health()
    
    def give_health(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.health += 1
            self.health -= 1
    
    def move(self) -> None:
        possible_moves = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_moves) #random move
        self.model.grid.move_agent(self, new_position) #call to move the agent
        print(f"Agent number {str(self.unique_id)}")
        print(f"Agent health {str(self.health)}")