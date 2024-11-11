import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from WorkerAgent import worker_agent

class factory_model(Model):
    """Factory environment where WorkerAgents interact."""
    def __init__(self, width, height, N, visualization=False):
        # Base model parameters
        self.num_agents = N
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.visualization = visualization

        # Quarantine 
        self.quarantine_zone = []
        self.quarantine_threshold = 12 # 12 steps of having an infection before 
        
        # Reinforcement learning parameters
        self.mask_mandate = False
        self.social_distancing = True
        self.num_vaccinated = 0
        self.splitting_level = 3 # 0 no grid splitting, 1 half, 2 quarter, 3 eights
        self.splitting_costs = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3}
        self.section_boundaries = []
        
        # Shift and day parameters
        self.steps_per_day = 24
        self.shifts_per_day = 3
        self.steps_per_shift = self.steps_per_day // self.shifts_per_day
        self.current_shift = 0
        self.next_shift_change = self.steps_per_shift

        #Stats gathering parameters
        self.productivity = 1.0
        self.current_reward = 0.0
        self.cumulative_reward = 0.0
        self.current_day = 0
        self.daily_infections = 0
        self.previous_infected_count = 0
        self.daily_stats = []
        self.daily_stats.extend([{
            'quarantined': 0,
            'returned_from_quarantine': 0
        }])

        self.initialize_agents()
        self.datacollector = DataCollector({
            "Healthy": lambda m: self.count_health_status("healthy"),
            "Infected": lambda m: self.count_health_status("infected"),
            "Recovered": lambda m: self.count_health_status("recovered"),
            "Productivity": lambda m: self.calculate_productivity(),
            "Quarantined": lambda m: len(self.quarantine_zone),
            "Total Reward": lambda m: self.current_reward,
            "Daily Infections": lambda m: self.daily_infections
        })

    def update_section_boundaries(self):
        """Update section boundaries based on current splitting level"""
        self.section_boundaries = []
        if self.splitting_level >= 1:
            self.section_boundaries.append(self.grid.width // 2)  # Half
        if self.splitting_level >= 2:
            self.section_boundaries.extend([self.grid.width // 4, 3 * self.grid.width // 4])  # Quarters
        if self.splitting_level >= 3:
            self.section_boundaries.extend([
                self.grid.width // 8,
                3 * self.grid.width // 8,
                5 * self.grid.width // 8,
                7 * self.grid.width // 8
            ])  # Eighths
        self.section_boundaries = sorted(list(set(self.section_boundaries))) 
    
    def get_section_index(self, x_coord):
        """Determine which section an x-coordinate falls into"""
        if self.splitting_level == 0:
            return 0
            
        section_width = self.grid.width // (2 ** self.splitting_level)
        return x_coord // section_width

    def get_section_bounds(self, section_idx):
        """Get the x-coordinate bounds for a given section"""
        if self.splitting_level == 0:
            return 0, self.grid.width
            
        section_width = self.grid.width // (2 ** self.splitting_level)
        start = section_idx * section_width
        end = start + section_width
        return start, end
    
    def is_position_clear(self, pos):
        """Check if the position is clear of other agents within a 1-cell radius."""
        if pos == None:
            return False
        neighbors = self.grid.get_neighbors(pos, moore=True, radius=1, include_center=False)
        return len(neighbors) == 0
    
    def process_shift_change(self):
        """Handle the shift change by redistributing workers within their sections"""
        self.current_shift = (self.current_shift + 1) % self.shifts_per_day
        num_sections = 2 ** self.splitting_level if self.splitting_level > 0 else 1
    
        agents_by_section = {i: [] for i in range(num_sections)}
    
        active_agents = [agent for agent in self.schedule.agents if not agent.is_quarantined]
    
        for agent in active_agents:
            section = self.get_section_index(agent.pos[0])
            agents_by_section[section].append(agent)
    
        for section_idx in range(num_sections):
            x_start, x_end = self.get_section_bounds(section_idx)
            section_agents = agents_by_section[section_idx]
        
            positions = [(x, y) for x in range(x_start, x_end) 
                            for y in range(self.grid.height)]
            self.random.shuffle(positions)
        
            for agent in section_agents:
                if positions:
                    if self.social_distancing:
                        new_pos = positions.pop()
                        if new_pos != agent.pos:
                            self.grid.move_agent(agent, new_pos)
                            agent.set_base_position(new_pos)
                            agent.steps_since_base_change = 0

        self.next_shift_change = (
            self.current_step_in_day + self.steps_per_shift
        ) % self.steps_per_day
        
    def return_from_quarantine(self, agent):
        """Return agent from quarantine to a random valid position"""
        if agent in self.quarantine_zone:
            if hasattr(agent, 'last_section'):
                section = agent.last_section
            else:
                section = self.random.randrange(2 ** self.splitting_level) if self.splitting_level > 0 else 0
        
            section_width = self.grid.width // (2 ** self.splitting_level) if self.splitting_level > 0 else self.grid.width
            x_start = section * section_width
            x_end = x_start + section_width
        
            new_x = self.random.randrange(x_start, x_end)
            new_y = self.random.randrange(self.grid.height)
        
            self.quarantine_zone.remove(agent)
            self.grid.place_agent(agent, (new_x, new_y))
            agent.is_quarantined = False
            agent.set_base_position((new_x, new_y))
            print(f"Agent {agent.unique_id} returned from quarantine to position ({new_x}, {new_y})")

    def quarantine_agent(self, agent):
        """Move agent to quarantine zone"""
        if agent not in self.quarantine_zone:
            if agent.pos is not None:
                agent.last_section = self.get_section_index(agent.pos[0])
            self.grid.remove_agent(agent)
            self.quarantine_zone.append(agent)
            agent.is_quarantined = True
            print(f"Agent {agent.unique_id} moved to quarantine")

    def process_quarantine(self):
        """Process quarantine state changes"""
        for agent in self.schedule.agents:
            if (agent.health_status == "infected" and 
                agent.infection_time > self.quarantine_threshold and 
                not agent.is_quarantined):
                self.quarantine_agent(agent)

        for agent in self.quarantine_zone.copy():
            if agent.health_status == "recovered":
                self.return_from_quarantine(agent)

    def initialize_agents(self):
        first_infections = random.randrange(self.num_agents)
        num_sections = 2 ** self.splitting_level if self.splitting_level > 0 else 1

        for i in range(self.num_agents):
            section = random.randrange(num_sections)
            worker = worker_agent(i, self, f'section_{section}')
            if i == first_infections:
                worker.health_status = "infected"

            self.schedule.add(worker)

            section_width = self.grid.width // num_sections
            x_start = section * section_width
            x_end = (section + 1) * section_width
            
            x = self.random.randrange(x_start, x_end)
            y = self.random.randrange(self.grid.height)
            
            pos = (x, y)
            self.grid.place_agent(worker, pos)
            worker.set_base_position(pos)
        self.previous_infected_count = self.count_health_status("infected")

    def step(self, action=None):
        """Execute one time step of the environment."""
        self.current_step_in_day = self.schedule.steps % self.steps_per_day
        
        action_cost = 0
        if not self.visualization and action is not None:
            action_cost = self.apply_action(action)
            if action == 3:
                self.update_section_boundaries()
        
        self.process_quarantine()

        if self.current_step_in_day == self.next_shift_change:
            self.process_shift_change()
            for agent in self.schedule.agents:
                if self.social_distancing and agent.pos is not None:
                    if self.schedule.steps % 3 == 0:  # Move only every 3 steps
                        possible_moves = self.grid.get_neighborhood(agent.pos, moore=True, radius=1)
                        valid_moves = [pos for pos in possible_moves if pos is not None and self.is_position_clear(pos)]
                        if valid_moves:
                            new_pos = random.choice(valid_moves)
                            self.grid.move_agent(agent, new_pos)
            else:
                self.schedule.step()

        current_infected = self.count_health_status("infected")
        new_infections = max(0, current_infected - self.previous_infected_count)
        self.daily_infections += new_infections
        self.previous_infected_count = current_infected

        if self.current_step_in_day == self.steps_per_day - 1:
            self.process_day_end()

        self.datacollector.collect(self)
        self.schedule.step()

        reward = self.calculate_reward() - action_cost
        self.current_reward = reward
        self.cumulative_reward += reward

        if not self.visualization:
            next_state = self.get_state()
            done = self.is_done()
            info = {
                'day': self.current_day,
                'daily_infections': self.daily_infections,
                'productivity': self.calculate_productivity(),
                'quarantined': len(self.quarantine_zone),
                'action_cost': action_cost
            }
            return next_state, reward, done, info

    
    def process_day_end(self):
        """Process the end of a day and store daily statistics"""
        self.current_day += 1
        self.daily_stats.append({
            'day': self.current_day,
            'infections': self.daily_infections,
            'healthy': self.count_health_status("healthy"),
            'infected': self.count_health_status("infected"),
            'recovered': self.count_health_status("recovered"),
            'productivity': self.calculate_productivity()
        })
        self.daily_infections = 0

    def visualization_step(self):
        """Advance the model one step without external actions (for visualization)."""
        current_infected = self.count_health_status("infected")
        new_infections = max(0, current_infected - self.previous_infected_count)
        self.daily_infections += new_infections
        self.previous_infected_count = current_infected

        if self.schedule.steps % self.steps_per_day == 0:
            self.process_day_end()

        reward = self.calculate_reward()
        self.current_reward += reward 
        self.current_productivity = self.calculate_productivity()
        
        self.datacollector.collect(self)
        self.schedule.step()

    def apply_action(self, action):
        action_cost = 0
        if action == 0:
            self.mask_mandate = not self.mask_mandate
            action_cost = 0.1 if self.mask_mandate else 0
        elif action == 1:
            self.social_distancing = not self.social_distancing
            action_cost = 0.15 if self.social_distancing else 0
        elif action == 2:
            before_vaccinated = self.num_vaccinated
            self.num_vaccinated = min(self.num_agents, self.num_vaccinated + 5)
            newly_vaccinated = self.num_vaccinated - before_vaccinated
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
        """Redistribute agents when changing splitting levels"""
        for agent in self.schedule.agents:
            section = self.get_section_index(agent.pos[0])
            num_sections = 2 ** self.splitting_level if self.splitting_level > 0 else 1
            section_width = self.grid.width // num_sections
            x_start = section * section_width
            x_end = (section + 1) * section_width
            
            new_x = self.random.randrange(x_start, x_end)
            new_y = self.random.randrange(self.grid.height)
            self.grid.move_agent(agent, (new_x, new_y))
            
    def get_state(self):
        healthy = self.count_health_status("healthy")
        infected = self.count_health_status("infected")
        recovered = self.count_health_status("recovered")
        return [healthy, infected, recovered, self.num_vaccinated]

    def calculate_reward(self):
        """Calculate reward with penalties for new infections"""
        working_agents = self.count_health_status("healthy") + self.count_health_status("recovered")
        base_reward = (working_agents / self.num_agents)
        
        infection_penalty = -0.5 * (self.daily_infections / self.num_agents)
        
        policy_penalty = -0.1 * (self.mask_mandate + self.social_distancing)
        vaccination_bonus = 0.05 * (self.num_vaccinated / self.num_agents)
        productivity = self.calculate_productivity() * 0.5
        return base_reward + infection_penalty + policy_penalty + vaccination_bonus + productivity
    
    def calculate_productivity(self):
        """Calculate current productivity based on various factors including recovered workers"""
        base_productivity = 1.0
        
        healthy_count = self.count_health_status("healthy")
        recovered_count = self.count_health_status("recovered")
        infected_count = self.count_health_status("infected")
        
        if self.mask_mandate:
            base_productivity *= 0.95  # 5% reduction for mask mandate
        if self.social_distancing:
            base_productivity *= 0.90  # 10% reduction for social distancing
        
        healthy_productivity = healthy_count
        recovered_productivity = recovered_count * 0.95  # Recovered work
        infected_productivity = infected_count * 0.2    # Infected workers at 20% productivity
        
        total_effective_workforce = (healthy_productivity + recovered_productivity + infected_productivity) / self.num_agents
        base_productivity *= total_effective_workforce
        
        vaccination_ratio = self.num_vaccinated / self.num_agents
        base_productivity *= (1 + 0.05 * vaccination_ratio)
        
        return base_productivity
    
    def count_health_status(self, status):
        return sum(1 for agent in self.schedule.agents if agent.health_status == status)
    def get_daily_stats(self):
        """Return the daily statistics"""
        return self.daily_stats
    def is_done(self):
        return self.count_health_status("infected") == 0 or self.schedule.steps > 100
