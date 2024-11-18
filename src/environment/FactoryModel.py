import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from WorkerAgent import worker_agent
from Quarantine import QuarantineManager
from config import FactoryConfig
from grid import GridManager
from Stats import StatsCollector
from Testing import TestingManager
from Testing import TestingPolicyHandler

class factory_model(Model):
    def __init__(self, width, height, N, visualization=False, config=None):
        super().__init__()
        if config is None:
            config = FactoryConfig(
                width=width,
                height=height,
                num_agents=N,
                visualization=visualization
            )
        # Base model parameters
        self.num_agents = config.num_agents
        self.grid = MultiGrid(config.width, config.height, torus=True)
        self.schedule = RandomActivation(self)
        self.visualization = config.visualization

        # Policy parameters
        self.mask_mandate = config.mask_mandate
        self.social_distancing = config.social_distancing
        self.num_vaccinated = 0
        self.initial_cleaning = config.cleaning_type
        self.test_lvl = config.testing_level
        self._splitting_level = config.splitting_level

        # Shift Parameters 
        self.steps_per_day = config.steps_per_day
        self.shifts_per_day = config.shifts_per_day
        self.steps_per_shift = config.steps_per_shift
        self.next_shift_change = self.steps_per_shift

        # Initialize managers
        self.quarantine = QuarantineManager(self)
        self.grid_manager = GridManager(self._splitting_level, self)
        self.stats = StatsCollector(self)
        self.testing = TestingManager(self)
        self.testing.set_testing_level(self.test_lvl)

        #Time tracking variables
        self.current_step = 0
        self.current_step_in_day = 0
        self.current_day = 0
        self.current_shift = 0 
            
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

    def initialize_datacollector(self):
        self.datacollector = DataCollector({
            "Healthy": lambda m: m.stats.count_health_status("healthy"),
            "Infected": lambda m: m.stats.count_health_status("infected"),
            "Recovered": lambda m: m.stats.count_health_status("recovered"),
            "Productivity": lambda m: (m.stats.calculate_productivity() * 
                                     m.testing.get_productivity_modifier() * 
                                     m.grid_manager.get_cleaning_productivity_modifier()),
            "Quarantined": lambda m: len(m.quarantine.quarantine_zone),
            "Daily Infections": lambda m: m.stats.daily_infections,
            "Current Shift": lambda m: m.current_shift,
            "Shifts Per Day": lambda m: m.shifts_per_day
        })

    
    def should_change_shift(self):
        """Check if shift change should occur"""
        return self.current_step_in_day == self.next_shift_change

    def step(self, action=None):
        self.current_step += 1
        self.current_step_in_day = self.current_step % self.steps_per_day
        
        if self.current_step_in_day == 0:
            self.current_day += 1
            
        self.process_scheduled_events()

        pre_step_infected = self.stats.count_health_status("infected")
        self._process_agent_steps()
        post_step_infected = self.stats.count_health_status("infected")
        
        new_infections = max(0, post_step_infected - pre_step_infected)
        self.stats.update_infections(new_infections)
        
        if self.current_step_in_day == self.steps_per_day - 1:
            self.stats.process_day_end()
            
        self.datacollector.collect(self)
        
        self.testing.current_productivity_impact = 0

        if not self.visualization:
            return self._get_step_results(new_infections, post_step_infected)
            
    def _process_agent_steps(self):
        for agent in self.schedule.agents:
            if self.social_distancing and agent.pos is not None:
                self.grid_manager.move_agent_social_distance(agent)
            agent.step()
    
    def process_scheduled_events(self):
        """Process all scheduled events in the correct order"""
        self.grid_manager.process_cleaning(self.current_step_in_day)
        for testing_type in ['light', 'medium', 'heavy']:
            if self.testing.should_run_testing(testing_type):
                self.testing.process_testing(testing_type)
            
        self.quarantine.process_quarantine()
        
        if self.should_change_shift():
            self.grid_manager.process_shift_change()
            self.next_shift_change = (self.current_step_in_day + self.steps_per_shift) % self.steps_per_day

    def get_steps_per_shift(self):
        return self.steps_per_shift
    
    def set__splitting_level(self, level):
        """Set the grid splitting level dynamically."""
        if level < 0 or level > 3:
            raise ValueError("Splitting level must be between 0 and 3.")
        self._splitting_level = level
        self.grid_manager.splitting_level = level
    
    def get__splitting_level(self):
        """Get the current splitting level."""
        return self._splitting_level
    
    @property
    def splitting_level(self):
        """Access splitting level through grid manager"""
        return self._splitting_level
    
    @splitting_level.setter
    def splitting_level(self, value):
        """Setter for splitting level that updates both factory and grid manager"""
        if not isinstance(value, int) or value < 0 or value > 3:
            raise ValueError("Splitting level must be an integer between 0 and 3")
        self._splitting_level = value
        if hasattr(self, 'grid_manager'):
            self.grid_manager.update_splitting_level(value)

    def _get_step_results(self, new_infections, total_infected):
        base_productivity = self.stats.calculate_productivity()
        cleaning_modifier = self.grid_manager.get_cleaning_productivity_modifier()
        testing_modifier = self.testing.get_productivity_modifier()
        
        return (
            self.stats.get_state(),
            self.stats.is_done(),
            {
                'day': self.stats.current_day,
                'step_in_day': self.current_step_in_day,
                'new_infections': new_infections,
                'total_infected': total_infected,
                'productivity': base_productivity * cleaning_modifier * testing_modifier,
                'quarantined': len(self.quarantine.quarantine_zone),
                'base_production': base_productivity * 2.0,
                'infection_penalty': -2.0 * (new_infections / self.num_agents)
            }
        )

    def set_splitting_level(self, value):
        """Public method to set splitting level"""
        self.splitting_level = value
    
    def get_splitting_level(self):
        """Public method to get splitting level"""
        return self.splitting_level
    
    def update_config(self, action_dict):
        """Update model configuration based on RL action"""
        config = FactoryConfig(
            cleaning_type=self.initial_cleaning,
            splitting_level=self._splitting_level,
            testing_level=self.test_lvl,
            social_distancing=self.social_distancing,
            mask_mandate=self.mask_mandate,
            shifts_per_day=self.shifts_per_day,
            steps_per_day=self.steps_per_day,
            visualization=self.visualization
        )
        
        old_shifts = config.shifts_per_day
        
        config.update_from_action(action_dict)
        
        self.initial_cleaning = config.cleaning_type
        self.grid_manager.set_cleaning_type(config.cleaning_type)
        
        self.splitting_level = config.splitting_level
        self.test_lvl = config.testing_level
        self.testing.set_testing_level(config.testing_level)
        
        self.social_distancing = config.social_distancing
        self.mask_mandate = config.mask_mandate
        
        if old_shifts != config.shifts_per_day:
            self.shifts_per_day = config.shifts_per_day
            self.steps_per_shift = config.steps_per_shift
            self.next_shift_change = (
                (self.current_step_in_day // self.steps_per_shift + 1) * 
                self.steps_per_shift
            ) % self.steps_per_day
            
            self.grid_manager.process_shift_change()