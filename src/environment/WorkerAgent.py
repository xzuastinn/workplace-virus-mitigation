from mesa import Agent
import random 

class WorkerAgent(Agent):
    """A worker agent with a health status and a preferred side of the factory."""
    def __init__(self, unique_id, model, side):
        super().__init__(unique_id, model)
        self.side = side
        self.health_status = "healthy"
        self.infection_time = 0
        self.had_covid = False  # Ensure this attribute is defined for all agents
    
    def move(self):
        """Generate possible moves then take a random action within the designated side of the factory."""
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
        """Spread infection based on proximity and infection history."""
        if self.health_status == "infected":
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for neighbor in cellmates:
                if isinstance(neighbor, WorkerAgent) and neighbor.health_status == "healthy":
                    # Determine infection probability based on whether the neighbor has had COVID
                    if neighbor.had_covid:
                        infection_probability = 0.25  # 25% chance if recovered
                    else:
                        infection_probability = 0.5  # 50% chance if never infected

                    # Apply infection with the calculated probability
                    if random.random() < infection_probability:
                        neighbor.health_status = "infected"
                        neighbor.had_covid = True  # Mark as having had COVID

    def update_health(self):
        """Update the health status of an infected worker over time."""
        if self.health_status == 'infected':
            self.infection_time += 1
            if self.infection_time > 14:
                self.health_status = 'recovered'

    def step(self):
        """Define agent's behavior per step."""
        self.move()
        self.infection()
        if self.health_status == "infected":
            self.infection_time += 1
            if self.infection_time > 14:
                self.health_status = "recovered"
