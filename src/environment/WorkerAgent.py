from mesa import Agent
import random 

class worker_agent(Agent):
    """A worker agent with a health status and a preferred side of the factory."""
    def __init__(self, unique_id, model, side):
        super().__init__(unique_id, model)
        self.side = side
        self.health_status = "healthy"
        self.infection_time = 0
        self.had_covid = False
        self.base_position = None
        self.steps_since_base_change = 0
        self.mask = False
        self.social_distance = 0 
    
    def set_base_position(self, pos):
        """Set the initial base position for the worker."""
        self.base_position = pos
        
    def get_valid_3x3_positions(self):
        """Get all valid positions within the 3x3 grid centered on base_position."""
        if not self.base_position:
            return []
            
        base_x, base_y = self.base_position
        possible_positions = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x = base_x + dx
                new_y = base_y + dy
                
                if (0 <= new_x < self.model.grid.width and 
                    0 <= new_y < self.model.grid.height):
                    
                    if ((self.side == 'left' and new_x < self.model.central_line) or
                        (self.side == 'right' and new_x >= self.model.central_line)):
                        possible_positions.append((new_x, new_y))
                        
        return possible_positions
        
    def update_base_position(self):
        """Update the base position after 24 steps."""
        if self.side == 'left':
            new_x = self.random.randrange(self.model.central_line)
        else:
            new_x = self.random.randrange(self.model.central_line, self.model.grid.width)
            
        new_y = self.random.randrange(self.model.grid.height)
        self.base_position = (new_x, new_y)
        self.steps_since_base_change = 0
        
        self.model.grid.move_agent(self, self.base_position)
    
    def move(self):
        """Move within the 3x3 grid around base position."""
        if self.base_position is None:
            self.set_base_position(self.pos)
            
        self.steps_since_base_change += 1
        if self.steps_since_base_change >= 24:
            self.update_base_position()
            return
            
        possible_moves = self.get_valid_3x3_positions()
        
        if possible_moves:
            new_position = self.random.choice(possible_moves)
            self.model.grid.move_agent(self, new_position)

    def get_infection_probability(self, distance, had_covid):
        """
        Calculate infection probability based on distance and immunity status.
        
        Args:
            distance (int): Distance from infected agent (0 = same cell, 1-3 = radius)
            had_covid (bool): Whether the potential infectee has had COVID before
        
        Returns:
            float: Probability of infection
        """
        base_probabilities = {
            0: 0.2,  # Same cell
            1: 0.1,  # Adjacent cells
            2: 0.05,  # Two cells away
            3: 0.01   # Three cells away
        }
        
        base_prob = base_probabilities.get(distance, 0)
        
        if had_covid:
            base_prob *= 0.5
            
        if self.model.mask_mandate:
            base_prob *= 0.7  # Masks reduce transmission by 30%
        if self.model.social_distancing:
            base_prob *= 0.8  # Social distancing reduces transmission by 20%
            
        return base_prob

    def get_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def infection(self):
        """Spread infection based on proximity with radius-based probability."""
        if self.health_status == "infected":
            x, y = self.pos
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    target_pos = (x + dx, y + dy)
                    
                    if (0 <= target_pos[0] < self.model.grid.width and 
                        0 <= target_pos[1] < self.model.grid.height):
                        
                        distance = self.get_manhattan_distance(self.pos, target_pos)
                        
                        if distance <= 3:
                            cell_agents = self.model.grid.get_cell_list_contents([target_pos])
                            
                            for agent in cell_agents:
                                if (isinstance(agent, worker_agent) and 
                                    agent.health_status == "healthy"):
                                    
                                    infection_prob = self.get_infection_probability(
                                        distance, 
                                        agent.had_covid
                                    )
                                    
                                    # Attempt infection
                                    if random.random() < infection_prob:
                                        agent.health_status = "infected"
                                        agent.had_covid = True

    def step(self):
        """Define agent's behavior per step."""
        self.move()
        self.infection()
        if self.health_status == "infected":
            self.infection_time += 1
            self.had_covid = True
            if self.infection_time > 30:
                self.health_status = "recovered"