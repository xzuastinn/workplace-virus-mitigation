import random


class GridManager:
    def __init__(self, model):
        self.model = model
        self.splitting_level = 1 # 0 full grid, 1 half, 2 quarter, 3 eights
        self.splitting_costs = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}
        self.section_boundaries = []
        self.cleaning_schedule = {
            'light': {'frequency': 8, 'infection_reduction': 0.2, 'duration': 1},
            'medium': {'frequency': 16, 'infection_reduction': 0.5, 'duration': 2},
            'heavy': {'frequency': 24, 'infection_reduction': 0.8, 'duration': 3}
        }
        self.current_cleaning = 'light'
        self.cleaning_steps_remaining = 0
        self.next_cleaning_steps = {
            'light': 8,
            'medium': 16,
            'heavy': 24
        }
        self.update_section_boundaries()
        self.section_infection_levels = [0] * (2 ** self.splitting_level if self.splitting_level > 0 else 1)
        
    def update_section_boundaries(self):
        self.section_boundaries = []
        if self.splitting_level >= 1:
            self.section_boundaries.append(self.model.grid.width // 2)
        if self.splitting_level >= 2:
            self.section_boundaries.extend([self.model.grid.width // 4, 3 * self.model.grid.width // 4])
        if self.splitting_level >= 3:
            self.section_boundaries.extend([
                self.model.grid.width // 8,
                3 * self.model.grid.width // 8,
                5 * self.model.grid.width // 8,
                7 * self.model.grid.width // 8
            ])
        self.section_boundaries = sorted(list(set(self.section_boundaries)))
        
        num_sections = 2 ** self.splitting_level if self.splitting_level > 0 else 1
        self.section_infection_levels = [0] * num_sections
    
    def get_section_for_agent(self, agent_id):
        num_sections = 2 ** self.splitting_level if self.splitting_level > 0 else 1
        return f'section_{agent_id // (self.model.num_agents // num_sections)}'
    
    def get_random_positions(self, num_positions):
        positions = [(x, y) for x in range(self.model.grid.width) 
                    for y in range(self.model.grid.height)]
        self.model.random.shuffle(positions)
        return positions[:num_positions]
        
    def get_section_index(self, x_coord):
        if self.splitting_level == 0:
            return 0
            
        num_sections = 2 ** self.splitting_level
        section_width = max(1, self.model.grid.width // num_sections)
        section_index = min(x_coord // section_width, num_sections - 1)
        return section_index
        
    def get_valid_position(self, agent):
        section = (getattr(agent, 'last_section', None) or 
                  self.model.random.randrange(2 ** self.splitting_level) if self.splitting_level > 0 else 0)
        
        section_width = (self.model.grid.width // (2 ** self.splitting_level) 
                        if self.splitting_level > 0 else self.model.grid.width)
        x_start = section * section_width
        
        new_x = self.model.random.randrange(x_start, x_start + section_width)
        new_y = self.model.random.randrange(self.model.grid.height)
        
        return (new_x, new_y)
        
    def move_agent_social_distance(self, agent):
        possible_moves = self.model.grid.get_neighborhood(agent.pos, moore=True, radius=1)
        valid_moves = [pos for pos in possible_moves if pos is not None]
        if valid_moves:
            new_pos = random.choice(valid_moves)
            self.model.grid.move_agent(agent, new_pos)
            
    def process_shift_change(self):
        self.model.current_shift = (self.model.current_shift + 1) % self.model.shifts_per_day

        active_agents = [agent for agent in self.model.schedule.agents if not agent.is_quarantined]
        occupied_positions = []

        for agent in active_agents:
            current_section_index = self.get_section_index(agent.pos[0])
            
            section_width = self.model.grid.width // (2 ** self.splitting_level if self.splitting_level > 0 else 1)
            section_start = current_section_index * section_width
            section_end = section_start + section_width

            attempts = 0
            max_attempts = 10
            placed = False

            while attempts < max_attempts and not placed:
                new_x = self.model.random.randrange(section_start, section_end)
                new_y = self.model.random.randrange(self.model.grid.height)
                new_pos = (new_x, new_y)

                if new_pos not in occupied_positions:
                    if new_pos != agent.pos:
                        self.model.grid.move_agent(agent, new_pos)
                        agent.set_base_position(new_pos)
                        agent.steps_since_base_change = 0
                        occupied_positions.append(new_pos)
                        agent.section = f'section_{current_section_index}'
                        agent.last_section = current_section_index
                        placed = True
                    break
                attempts += 1

            if not placed:
                self.model.grid.remove_agent(agent)
                self.model.schedule.remove(agent)

        self.model.next_shift_change = ((self.model.current_step_in_day + self.model.steps_per_shift) % self.model.steps_per_day)
                                        
    def apply_action(self, action):
        action_cost = 0
        if action == 0:
            self.model.mask_mandate = not self.model.mask_mandate
            action_cost = 0.1 if self.model.mask_mandate else 0
        elif action == 1:
            self.model.social_distancing = not self.model.social_distancing
            action_cost = 0.15 if self.model.social_distancing else 0
        elif action == 2:
            before_vaccinated = self.model.num_vaccinated
            self.model.num_vaccinated = min(self.model.num_agents, 
                                          self.model.num_vaccinated + 5)
            newly_vaccinated = self.model.num_vaccinated - before_vaccinated
            action_cost = 0.05 * newly_vaccinated
        elif action == 3:
            old_level = self.splitting_level
            self.splitting_level = (self.splitting_level + 1) % 4
            self.update_section_boundaries()
            action_cost = self.splitting_costs[self.splitting_level]
            
            if old_level > self.splitting_level:
                self.redistribute_agents()
        return action_cost
        
    def redistribute_agents(self):
        for agent in self.model.schedule.agents:
            if agent.pos is not None:
                agent.section = self.get_section_for_agent(agent.unique_id)
                new_pos = self.get_valid_position(agent)
                self.model.grid.move_agent(agent, new_pos)
    
    def update_infection_level(self, section_index, infected_count):
        num_sections = len(self.section_infection_levels)
        if 0 <= section_index < num_sections:
            self.section_infection_levels[section_index] += infected_count
        
    def get_infection_probability(self, section_index):
        base_probability = 0.1
        balancer = 0.5
        return base_probability * (self.section_infection_levels[section_index] ** balancer)
        
    def process_cleaning(self):
        """Process cleaning activities and their effects"""
        if self.cleaning_steps_remaining > 0:
            self.apply_cleaning_effects()
            self.cleaning_steps_remaining -= 1
            if self.cleaning_steps_remaining == 0:
                self.current_cleaning = None
        else:
            for cleaning_type, next_step in self.next_cleaning_steps.items():
                if self.model.current_step_in_day == next_step:
                    self.start_cleaning(cleaning_type)
                    break

    def start_cleaning(self, cleaning_type):
        """Start a new cleaning cycle"""
        self.current_cleaning = cleaning_type
        self.cleaning_steps_remaining = self.cleaning_schedule[cleaning_type]['duration']
        
        self.next_cleaning_steps[cleaning_type] = (
            (self.model.current_step_in_day + self.cleaning_schedule[cleaning_type]['frequency']) 
            % self.model.steps_per_day
        )
        
        self.apply_cleaning_effects()

    def apply_cleaning_effects(self):
        """Apply the effects of current cleaning"""
        if self.current_cleaning:
            reduction = self.cleaning_schedule[self.current_cleaning]['infection_reduction']
            for i in range(len(self.section_infection_levels)):
                self.section_infection_levels[i] *= (1 - reduction)

    