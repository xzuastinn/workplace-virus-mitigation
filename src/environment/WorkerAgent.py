from mesa import Agent
import random

class worker_agent(Agent):
    """A worker agent with a health status and assigned section."""
    def __init__(self, unique_id, model, section):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        self.section = section
        self.health_status = "healthy"
        self.infection_time = 0
        self.recovery_time = 0
        self.had_covid = False
        self.is_quarantined = False
        self.base_production = random.uniform(0.9, 1.1)
        self.current_production = self.base_production
        self.confined_to_2x2 = False
        self.confined_steps = 0
        self.base_position = None
        self.steps_since_base_change = 0

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
        self.confined_to_2x2 = True
        self.confined_steps = 0

    def get_valid_positions(self):
        """Get all valid positions on the grid within the agent's section"""
        x_start, x_end = self.get_section_bounds()
        
        if self.base_position is None:
            self.set_base_position((x_start, 0))
            
        if self.confined_to_2x2:
            x, y = self.base_position
            potential_positions = [
                (x+dx, y+dy) for dx in range(0, 2) for dy in range(0, 2)
                if (x_start <= x+dx < x_end and
                    0 <= y+dy < self.model.grid.height)
            ]
        else:
            potential_positions = [(x, y) for x in range(x_start, x_end, 2) 
                                for y in range(self.model.grid.height) if y % 2 == 0]
        
        valid_positions = []
        for pos in potential_positions:
            cell_contents = self.model.grid.get_cell_list_contents([pos])
            if not cell_contents or (len(cell_contents) == 1 and cell_contents[0] == self):
                valid_positions.append(pos)
        
        return valid_positions
    
    def update_base_position(self):
        """Update the base position after n steps."""
        x_start, x_end = self.get_section_bounds()
        new_x = self.random.randrange(x_start, x_end, 2)
        new_y = self.random.randrange(0, self.model.grid.height, 2)
        self.base_position = (new_x, new_y)
        self.steps_since_base_change = 0
        self.confined_to_2x2 = True
        self.confined_steps = 0

        self.model.grid.move_agent(self, self.base_position)

    def move(self):
        """Move the agent to a random valid position on the grid."""
        if self.is_quarantined:
            return

        if self.base_position is None:
            self.set_base_position(self.pos)

        self.steps_since_base_change += 1

        if self.steps_since_base_change > self.model.get_steps_per_shift():
            self.update_base_position()
            return

        if self.model.social_distancing:
            valid_positions = []
            check_positions = self.get_valid_positions()
            MINIMUM_DISTANCE = 1
            
            for pos in check_positions:
                is_valid = True
                for neighbor_pos in self.model.grid.iter_neighborhood(
                    pos, moore=True, radius=MINIMUM_DISTANCE
                ):
                    if (0 <= neighbor_pos[0] < self.model.grid.width and 
                        0 <= neighbor_pos[1] < self.model.grid.height):
                        cell_contents = self.model.grid.get_cell_list_contents([neighbor_pos])
                        if any(isinstance(a, type(self)) for a in cell_contents):
                            is_valid = False
                            break
                
                if is_valid:
                    valid_positions.append(pos)
            
            if not valid_positions:
                min_neighbors = float('inf')
                best_positions = []
                
                for pos in check_positions:
                    neighbor_count = sum(
                        1 for neighbor_pos in self.model.grid.iter_neighborhood(
                            pos, moore=True, radius=MINIMUM_DISTANCE
                        )
                        if (0 <= neighbor_pos[0] < self.model.grid.width and 
                            0 <= neighbor_pos[1] < self.model.grid.height and
                            any(isinstance(a, type(self)) 
                                for a in self.model.grid.get_cell_list_contents([neighbor_pos]))
                        )
                    )
                    
                    if neighbor_count < min_neighbors:
                        min_neighbors = neighbor_count
                        best_positions = [pos]
                    elif neighbor_count == min_neighbors:
                        best_positions.append(pos)
                
                valid_positions = best_positions
        else:
            valid_positions = self.get_valid_positions()
        
        if valid_positions:
            new_position = random.choice(valid_positions)
            self.model.grid.move_agent(self, new_position)


    def get_infection_probability(self, distance, had_covid):
        """Calculate infection probability based on distance and immunity status."""
        base_probabilities = {
            0: 0.4,  # Same cell
            1: 0.12,  # Adjacent cells
            2: 0.08,  # Two cells away
            3: 0.05   # Three cells away
        }
        
        base_prob = base_probabilities.get(distance, 0)
        section_index = self.model.grid_manager.get_section_index(self.pos[0])

        target_section_index = self.model.grid_manager.get_section_index(self.pos[0])
        if section_index != target_section_index:
            base_prob *= 0.15

        section_prob = self.model.grid_manager.get_infection_probability(section_index)
        if had_covid:
            base_prob *= 0.5
            
        if self.model.mask_mandate:
            base_prob *= 0.7  # Masks reduce transmission by 30%
        if self.model.social_distancing:
            base_prob *= 0.8  # Social distancing reduces transmission by 20%
            
        return base_prob * section_prob
    
    def update_infection(self):
        if self.health_status == "infected":
            self.infection_time += 1
            self.had_covid = True
            if self.infection_time > 40:
                self.health_status = "recovered"
        elif self.health_status == "recovered":
            self.infection_time += 1
            if self.infection_time > 100:
                self.health_status = "healthy"
                self.infection_time = 0

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
            section_index = self.model.grid_manager.get_section_index(self.pos[0])
            self.model.grid_manager.update_infection_level(section_index, 1 if self.health_status == "infected" else 0)
            
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
                                    
                                    target_section = self.model.grid_manager.get_section_index(target_pos[0])
                                    infection_prob = self.get_infection_probability(
                                        distance, 
                                        agent.had_covid
                                    )
                                    
                                    if random.random() < infection_prob:
                                        agent.health_status = "infected"
                                        agent.had_covid = True
                                        self.model.grid_manager.update_infection_level(target_section, 1)

    def step(self):
        """Define agent's behavior per step."""
        if not self.is_quarantined:
            self.move()
            self.infection()
        self.update_infection()
        self.update_production()
