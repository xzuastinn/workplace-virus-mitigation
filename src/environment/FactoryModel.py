import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from WorkerAgent import worker_agent
from Quarantine import QuarantineManager
from grid import GridManager
from Stats import StatsCollector

class factory_model(Model):
    def __init__(self, width, height, N, visualization=False):
        super().__init__()
        # Base model parameters
        self.num_agents = N
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.visualization = visualization

        # Initialize managers
        self.quarantine = QuarantineManager(self)
        self.grid_manager = GridManager(self)
        self.stats = StatsCollector(self)
        
        # Policy parameters
        self.mask_mandate = False
        self.social_distancing = True
        self.num_vaccinated = 0
        
        # Time parameters
        self.steps_per_day = 24
        self.shifts_per_day = 3
        self.steps_per_shift = self.steps_per_day // self.shifts_per_day
        self.next_shift_change = self.steps_per_shift
        
        self.initialize_agents()
        self.initialize_datacollector()
        
    def initialize_agents(self):
        first_infection = random.randrange(self.num_agents)
        positions = self.grid_manager.get_random_positions(self.num_agents)
        num_sections = 2 ** self.grid_manager.splitting_level if self.grid_manager.splitting_level > 0 else 1

        for i in range(self.num_agents):
            section = f'section_{i // (self.num_agents // num_sections)}'
            worker = worker_agent(i, self, section)

            if i == first_infection:
                worker.health_status = "infected"

            self.schedule.add(worker)
            pos = positions[i]
            self.grid.place_agent(worker, pos)
            worker.set_base_position(pos)

    def initialize_datacollector(self):
        self.datacollector = DataCollector({
            "Healthy": lambda m: m.stats.count_health_status("healthy"),
            "Infected": lambda m: m.stats.count_health_status("infected"),
            "Recovered": lambda m: m.stats.count_health_status("recovered"),
            "Productivity": lambda m: m.stats.calculate_productivity(),
            "Quarantined": lambda m: len(m.quarantine.quarantine_zone),
            "Daily Infections": lambda m: m.stats.daily_infections
        })


    def step(self, action=None):
        self.current_step_in_day = self.schedule.steps % self.steps_per_day
        
        if not self.visualization and action is not None:
            action_cost = self.grid_manager.apply_action(action)
        else:
            action_cost = 0
            
        self.quarantine.process_quarantine()
        
        if self.current_step_in_day == self.next_shift_change:
            self.grid_manager.process_shift_change()

        pre_step_infected = self.stats.count_health_status("infected")
        self._process_agent_steps()
        post_step_infected = self.stats.count_health_status("infected")
        
        new_infections = max(0, post_step_infected - pre_step_infected)
        self.stats.update_infections(new_infections)
        
        if self.current_step_in_day == self.steps_per_day - 1:
            self.stats.process_day_end()
            
        self.datacollector.collect(self)
        
        if not self.visualization:
            return self._get_step_results(new_infections, action_cost, post_step_infected)
            
    def _process_agent_steps(self):
        for agent in self.schedule.agents:
            if self.social_distancing and agent.pos is not None:
                self.grid_manager.move_agent_social_distance(agent)
            agent.step()
    
    def get_steps_per_shift(self):
        return self.steps_per_shift
    
    @property
    def splitting_level(self):
        """Access splitting level through grid manager"""
        return self.grid_manager.splitting_level
    
    def _get_step_results(self, new_infections, action_cost, total_infected):
        
        return (
            self.stats.get_state(),
            self.stats.is_done(),
            {
                'day': self.stats.current_day,
                'step_in_day': self.current_step_in_day,
                'new_infections': new_infections,
                'total_infected': total_infected,
                'productivity': self.stats.calculate_productivity(),
                'quarantined': len(self.quarantine.quarantine_zone),
                'action_cost': action_cost,
                'base_production': self.stats.calculate_productivity() * 2.0,
                'infection_penalty': -2.0 * (new_infections / self.num_agents)
            }
        )