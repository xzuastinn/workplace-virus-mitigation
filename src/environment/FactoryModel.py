import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from WorkerAgent import worker_agent
from Quarantine import QuarantineManager
from grid import GridManager
from Stats import StatsCollector
from Testing import TestingManager

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
        self.testing = TestingManager(self)
        
        # Policy parameters
        self.mask_mandate = False
        self.social_distancing = False
        self.num_vaccinated = 0
        self.test_frequency = 20
        self.testing_start_step = 1
        self.next_test_step = self.testing_start_step 

        # Time parameters
        self.steps_per_day = 24
        self.shifts_per_day = 3
        self.steps_per_shift = self.steps_per_day // self.shifts_per_day
        self.next_shift_change = self.steps_per_shift
        self.current_step = 0
        self.current_step_in_day = 0
        self.current_day = 0
        self.current_shift = 0 

        # Testing schedule
        self.test_frequency = 20
        self.last_test_step = 0
        self.next_test_step = self.test_frequency
        
        # Shift change schedule
        self.next_shift_change = self.steps_per_shift
        
        # Cleaning schedule initialization
        self.initialize_cleaning_schedule()
        
        self.initialize_agents()
        self.initialize_datacollector()
        
    def initialize_agents(self):
        first_infection = random.randrange(self.num_agents)
        positions = self.grid_manager.get_random_positions(self.num_agents)
        num_sections = 2 ** self.grid_manager.splitting_level if self.grid_manager.splitting_level > 0 else 1

        for i in range(self.num_agents):
            section_index = positions[i][0] // (self.grid.width // num_sections)
            section = f'section_{section_index}'
            worker = worker_agent(i, self, section)

            if i == first_infection:
                worker.health_status = "infected"

            self.schedule.add(worker)
            pos = positions[i]
            self.grid.place_agent(worker, pos)
            worker.set_base_position(pos)
            worker.last_section = section_index

    def initialize_cleaning_schedule(self):
        """Initialize the cleaning schedule with proper intervals"""
        self.cleaning_schedules = {
            'light': {'next_step': 8, 'frequency': 8},
            'medium': {'next_step': 16, 'frequency': 16},
            'heavy': {'next_step': 24, 'frequency': 24}
        }

    def initialize_datacollector(self):
        self.datacollector = DataCollector({
            "Healthy": lambda m: m.stats.count_health_status("healthy"),
            "Infected": lambda m: m.stats.count_health_status("infected"),
            "Recovered": lambda m: m.stats.count_health_status("recovered"),
            "Productivity": lambda m: m.stats.calculate_productivity(),
            "Quarantined": lambda m: len(m.quarantine.quarantine_zone),
            "Daily Infections": lambda m: m.stats.daily_infections
        })

    def should_run_testing(self):
        """Improved testing schedule check"""
        if self.current_step >= self.next_test_step:
            self.last_test_step = self.current_step
            self.next_test_step = self.current_step + self.test_frequency
            return True
        return False
    
    def should_run_cleaning(self, cleaning_type):
        """Check if cleaning should be performed"""
        schedule = self.cleaning_schedules[cleaning_type]
        if self.current_step_in_day == schedule['next_step']:
            schedule['next_step'] = (self.current_step_in_day + schedule['frequency']) % self.steps_per_day
            return True
        return False
    
    def should_change_shift(self):
        """Check if shift change should occur"""
        return self.current_step_in_day == self.next_shift_change

    def step(self, action=None):
        self.current_step += 1
        self.current_step_in_day = self.current_step % self.steps_per_day
        
        if self.current_step_in_day == 0:
            self.current_day += 1
            
        if not self.visualization and action is not None:
            action_cost = self.grid_manager.apply_action(action)
        else:
            action_cost = 0
            
        self.process_scheduled_events()
        
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
    
    def process_scheduled_events(self):
        """Process all scheduled events in the correct order"""
        for cleaning_type in ['light', 'medium', 'heavy']:
            if self.should_run_cleaning(cleaning_type):
                self.grid_manager.start_cleaning(cleaning_type)
        
        if self.grid_manager.current_cleaning:
            self.grid_manager.process_cleaning()
            
        if self.should_run_testing():
            print(f"Running factory tests at step {self.current_step}")
            self.testing.process_testing()
            
        self.quarantine.process_quarantine()
        
        if self.should_change_shift():
            self.grid_manager.process_shift_change()
            self.next_shift_change = (self.current_step_in_day + self.steps_per_shift) % self.steps_per_day

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