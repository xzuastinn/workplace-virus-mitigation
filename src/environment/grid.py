import random


class GridManager:
    def __init__(self, model):
        self.model = model
        self.splitting_level = 2
        self.splitting_costs = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}
        self.section_boundaries = []
        self.update_section_boundaries()
        
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
        section_width = self.model.grid.width // (2 ** self.splitting_level)
        return x_coord // section_width
        
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
        
        active_agents = [agent for agent in self.model.schedule.agents 
                        if not agent.is_quarantined]
        positions = self.get_random_positions(len(active_agents))
        
        for agent, new_pos in zip(active_agents, positions):
            if new_pos != agent.pos:
                self.model.grid.move_agent(agent, new_pos)
                agent.set_base_position(new_pos)
                agent.steps_since_base_change = 0
                
        self.model.next_shift_change = ((self.model.current_step_in_day + 
                                       self.model.steps_per_shift) % 
                                      self.model.steps_per_day)
                                      
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
                # Update agent's section when redistributing
                agent.section = self.get_section_for_agent(agent.unique_id)
                new_pos = self.get_valid_position(agent)
                self.model.grid.move_agent(agent, new_pos)