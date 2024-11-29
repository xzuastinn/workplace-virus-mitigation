import random

class GridManager:
    """Class that handles the "factory floor" and agent movement within this area"""
    def __init__(self, initial_splitting_level, model):
        self.model = model
        self._splitting_level = initial_splitting_level # 0 full grid, 1 half, 2 quarter, 3 eights
        self.section_boundaries = []
        self.cleaning_schedule = {
            'light': {
                'frequency': 8, 
                'infection_reduction': 0.35, 
                'duration': 1,
                'production_reduction': 0.00
            },
            'medium': {
                'frequency': 16, 
                'infection_reduction': 0.65, 
                'duration': 2,
                'production_reduction': 0.15  # 15% production reduction
            },
            'heavy': {
                'frequency': 16, 
                'infection_reduction': 0.8, 
                'duration': 2,
                'production_reduction': 0.25  # 25% production reduction
            }
        }
        self.current_cleaning = self.model.initial_cleaning #initializes cleaning schedule
        self.cleaning_steps_remaining = 0
        self.next_cleaning = { #dictionary for step intervals
            'light': 8,
            'medium': 16,
            'heavy': 16
        }
        self.update_section_boundaries()
        self.section_infection_levels = [0] * (2 ** self._splitting_level if self._splitting_level > 0 else 1)
        
    def update_section_boundaries(self):
        """Creates the section boundaries based on the current splitting level"""
        self.section_boundaries = []
        if self._splitting_level >= 1:
            self.section_boundaries.append(self.model.grid.width // 2)
        if self._splitting_level >= 2:
            self.section_boundaries.extend([self.model.grid.width // 4, 3 * self.model.grid.width // 4])
        if self._splitting_level >= 3:
            self.section_boundaries.extend([
                self.model.grid.width // 8,
                3 * self.model.grid.width // 8,
                5 * self.model.grid.width // 8,
                7 * self.model.grid.width // 8
            ])
        self.section_boundaries = sorted(list(set(self.section_boundaries)))
        
        num_sections = 2 ** self._splitting_level if self._splitting_level > 0 else 1
        self.section_infection_levels = [0] * num_sections
    
    def get_section_for_agent(self, agent_id):
        """Helper method to get the current section of a provided agent"""
        num_sections = 2 ** self._splitting_level if self._splitting_level > 0 else 1
        return f'section_{agent_id // (self.model.num_agents // num_sections)}'
    
    def get_random_positions(self, num_positions):
        """Helper method for factory initializing agents to get random positions for agents to start in"""
        positions = [(x, y) for x in range(self.model.grid.width) 
                    for y in range(self.model.grid.height)]
        self.model.random.shuffle(positions)
        return positions[:num_positions]
        
    def get_section_index(self, x_coord):
        """Gets the section index of a provided x coordinate point"""
        if self._splitting_level == 0:
            return 0
            
        num_sections = 2 ** self._splitting_level
        section_width = max(1, self.model.grid.width // num_sections)
        section_index = min(x_coord // section_width, num_sections - 1)
        return section_index
    
    def get_valid_position(self, agent):
        """Helper method to get all the valid positions within a section for an agent to move to"""
        section = (getattr(agent, 'last_section', None) or 
                  self.model.random.randrange(2 ** self._splitting_level) if self._splitting_level > 0 else 0)
        
        section_width = (self.model.grid.width // (2 ** self._splitting_level) 
                        if self._splitting_level > 0 else self.model.grid.width)
        x_start = section * section_width
        
        new_x = self.model.random.randrange(x_start, x_start + section_width)
        new_y = self.model.random.randrange(self.model.grid.height)
        
        return (new_x, new_y)
        
    def move_agent_social_distance(self, agent):
        """Moves agent atleast 1 block away from other agents to start a shift"""
        possible_moves = self.model.grid.get_neighborhood(agent.pos, moore=True, radius=1)
        valid_moves = [pos for pos in possible_moves if pos is not None]
        if valid_moves:
            new_pos = random.choice(valid_moves)
            self.model.grid.move_agent(agent, new_pos)
            
    def process_shift_change(self):
        """Function to manage how a shift change is ran."""
        self.model.current_shift = (self.model.current_shift + 1) % self.model.shifts_per_day

        active_agents = [agent for agent in self.model.schedule.agents if agent.on_shift]
        inactive_agents = [agent for agent in self.model.schedule.agents if not agent.on_shift]

        for agent in active_agents[:]:
            if agent.pos is not None:
                self.model.grid.remove_agent(agent)
            agent.on_shift = False
            agent.pos = None
            self.model.active_agents.remove(agent)
            self.model.inactive_agents.append(agent)

        occupied_positions = set()
        num_sections = 2 ** self._splitting_level if self._splitting_level > 0 else 1
        agents_per_section = len(inactive_agents) // num_sections
        remainder = len(inactive_agents) % num_sections

        for section_index in range(num_sections):
            section_width = self.model.grid.width // num_sections
            section_start = section_index * section_width
            section_end = section_start + section_width

            section_agents = inactive_agents[
                section_index * agents_per_section + min(section_index, remainder):
                (section_index + 1) * agents_per_section + min(section_index + 1, remainder)
            ]

            for agent in section_agents:
                attempts = 0
                max_attempts = 50
                placed = False

                while attempts < max_attempts and not placed:
                    new_x = self.model.random.randrange(section_start, section_end)
                    new_y = self.model.random.randrange(self.model.grid.height)
                    new_pos = (new_x, new_y)

                    if new_pos not in occupied_positions:
                        self.model.grid.place_agent(agent, new_pos)
                        agent.set_base_position(new_pos)
                        agent.steps_since_base_change = 0
                        agent.on_shift = True
                        occupied_positions.add(new_pos)
                        agent.section = f'section_{section_index}'
                        agent.last_section = section_index

                        self.model.inactive_agents.remove(agent)
                        self.model.active_agents.append(agent)

                        placed = True
                    attempts += 1

                if not placed:
                    self.model.schedule.remove(agent)
                    self.model.inactive_agents.remove(agent)

        self.model.next_shift_change = ((self.model.current_step_in_day + self.model.steps_per_shift) % self.model.steps_per_day)
        
    def redistribute_agents(self):
        """redistributes agents to new sections when an update for section is called"""
        for agent in self.model.schedule.agents:
            if agent.pos is not None:
                agent.section = self.get_section_for_agent(agent.unique_id)
                new_pos = self.get_valid_position(agent)
                self.model.grid.move_agent(agent, new_pos)
    
    def update_infection_level(self, section_index, infected_count):
        """Update infection levels for a section based on infected count in the section index"""
        num_sections = len(self.section_infection_levels)
        if 0 <= section_index < num_sections:
            self.section_infection_levels[section_index] = min(
                self.section_infection_levels[section_index] + infected_count,
                10
            )
        
    def get_infection_probability(self, section_index):
        """Calculate infection probability based on section infection levels"""
        base_probability = 0.8
        infection_level = self.section_infection_levels[section_index]
        
        multiplier = min(1.0 + (infection_level * 0.1), 2.0) #capped at 2
        
        return base_probability * multiplier
            
    def process_cleaning(self, current_step_in_day):
        """Main cleaning logic processing method"""
        if self.cleaning_steps_remaining > 0:
            self.apply_cleaning_effects()
            self.cleaning_steps_remaining -= 1
            return

        cleaning_type = self.current_cleaning
        if current_step_in_day == self.next_cleaning[cleaning_type]: #checks if we are due for a cleaning
            print(f"Cleaning triggered: {cleaning_type} at step in day {current_step_in_day}")
            self.start_cleaning(cleaning_type) #call the cleaning method.
            self.next_cleaning[cleaning_type] = (
                (current_step_in_day + self.cleaning_schedule[cleaning_type]['frequency'])
                % self.model.steps_per_day
            )

    def start_cleaning(self, cleaning_type):
        """Start a new cleaning cycle"""
        self.current_cleaning = cleaning_type
        self.cleaning_steps_remaining = self.cleaning_schedule[cleaning_type]['duration']
        self.apply_cleaning_effects()

    def apply_cleaning_effects(self):
        """Apply the effects of current cleaning to reduce infection probability in the section"""
        if not self.current_cleaning:
            return
            
        schedule = self.cleaning_schedule[self.current_cleaning]
        
        reduction = schedule['infection_reduction']
        for i in range(len(self.section_infection_levels)):
            self.section_infection_levels[i] *= (1 - reduction)

    def set_cleaning_type(self, cleaning_type):
        """Change the cleaning type"""
        if cleaning_type in self.cleaning_schedule:
            self.current_cleaning = cleaning_type
            self.cleaning_steps_remaining = 0
    
    def get_cleaning_productivity_modifier(self):
        """Calculate productivity modifier based on current cleaning status"""
        if self.cleaning_steps_remaining > 0 and self.current_cleaning:
            return 1 - self.cleaning_schedule[self.current_cleaning]['production_reduction']
        return 1.0
    
    @property
    def splitting_level(self):
        """Get current splitting level"""
        return self._splitting_level

    def update_splitting_level(self, value):
        """Helper method to update splitting level and related configurations"""
        if self._splitting_level != value:
            self._splitting_level = value
            self.update_section_boundaries()
            num_sections = 2 ** self._splitting_level if self._splitting_level > 0 else 1
            self.section_infection_levels = [0] * num_sections
            if hasattr(self.model, 'schedule') and self.model.schedule is not None:
                self.redistribute_agents()