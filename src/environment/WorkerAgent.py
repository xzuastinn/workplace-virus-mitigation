from mesa import Agent
import random

class worker_agent(Agent):
    """A worker agent with a health status and assigned section."""
    def __init__(self, unique_id, model, section):
        super().__init__(unique_id, model)
        self.section = section
        self.health_status = "healthy"
        self.infection_time = 0
        self.had_covid = False
        self.base_position = None
        self.steps_since_base_change = 0
        self.mask = False
        self.social_distance = 0 
        self.is_quarantined = False
        self.base_production = random.uniform(0.9, 1.1)
        self.current_production = self.base_production

    
    def get_section_bounds(self):
        """Get the boundaries of the agent's assigned section"""
        section_num = int(self.section.split('_')[1])
        num_sections = 2 ** self.model.splitting_level if self.model.splitting_level > 0 else 1
        section_width = self.model.grid.width // num_sections
        x_start = section_num * section_width
        x_end = (section_num + 1) * section_width
        return x_start, x_end
    
    def set_base_position(self, pos):
        """Set the initial base position for the worker."""
        self.base_position = pos
        
    def get_valid_3x3_positions(self):
        """Get all valid positions within the 3x3 grid centered on base_position."""
        if not self.base_position:
            return []
            
        base_x, base_y = self.base_position
        possible_positions = []
        x_start, x_end = self.get_section_bounds()
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x = base_x + dx
                new_y = base_y + dy
                
                if (0 <= new_x < self.model.grid.width and 
                    0 <= new_y < self.model.grid.height and
                    x_start <= new_x < x_end):
                    possible_positions.append((new_x, new_y))
                        
        return possible_positions
        
    def update_base_position(self):
        """Update the base position after n steps."""
        x_start, x_end = self.get_section_bounds()
        new_x = self.random.randrange(x_start, x_end)
        new_y = self.random.randrange(self.model.grid.height)
        self.base_position = (new_x, new_y)
        self.steps_since_base_change = 0
        
        self.model.grid.move_agent(self, self.base_position)
    
    def move(self):
        """Move within the 3x3 grid around base position."""
        if self.is_quarantined:
            return
        
        if self.base_position is None:
            self.set_base_position(self.pos)
            
        self.steps_since_base_change += 1
            
        possible_moves = self.get_valid_3x3_positions()
        
        if possible_moves:
            new_position = self.random.choice(possible_moves)
            self.model.grid.move_agent(self, new_position)

    def get_infection_probability(self, distance, had_covid):
        """Calculate infection probability based on distance and immunity status."""
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
    
    def update_production(self):
        """Update agent's current production based on various factors."""
        production = self.base_production

        if self.health_status == "healthy":
            production = self.base_production
        elif self.health_status == "infected":
            production *= 0.2
        elif self.health_status == "recovered":
            production *= 0.95 

        if self.model.mask_mandate:
            production *= 0.95 
        if self.model.social_distancing:
            production *= 0.90  

        if self.unique_id < self.model.num_vaccinated:
            production *= 1.05 

        if self.is_quarantined:
            production = 0

        self.current_production = production

    def get_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def infection(self):
        """Spread infection based on proximity with radius-based probability."""
        if self.health_status == "infected":
            x, y = self.pos
            x_start, x_end = self.get_section_bounds()
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    target_pos = (x + dx, y + dy)
                    
                    if (0 <= target_pos[0] < self.model.grid.width and 
                        0 <= target_pos[1] < self.model.grid.height and
                        x_start <= target_pos[0] < x_end):
                        
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
                                    
                                    if random.random() < infection_prob:
                                        agent.health_status = "infected"
                                        agent.had_covid = True

    def step(self):
        """Define agent's behavior per step."""
        if not self.is_quarantined:
            self.move()
            self.infection()
        if self.health_status == "infected":
            self.infection_time += 1
            self.had_covid = True
            if self.infection_time > 30:
                self.health_status = "recovered"
                if self.infection_time > 100: # allow for reinfections
                    self.health_status = "healthy"
                    self.infection_time = 0
        self.update_production()
