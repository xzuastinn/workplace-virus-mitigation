import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))



import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from src.environment.WorkerAgent import worker_agent
from src.environment.infection_control.Quarantine import QuarantineManager
from src.environment.FactoryConfig import FactoryConfig
from src.environment.GridManager import GridManager
from src.environment.Stats import StatsCollector
from src.environment.infection_control.SwabTesting import TestingManager

class factory_model(Model):
    """Main class model that sets up the environment with provided parameters and agents"""
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

        # Policy parameters for RL Training
        self.mask_mandate = config.mask_mandate
        self.social_distancing = config.social_distancing
        self.initial_cleaning = config.cleaning_type
        self.test_lvl = config.testing_level
        self._splitting_level = config.splitting_level

        # Shift Parameters for RL training
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

        # counters
        self.swab_testing_counter = {"none": 0, "light": 0, "medium": 0, "heavy": 0} # done
        self.cleaning_counter = {"light": 0, "medium": 0, "heavy": 0}
        self.shifts_counter = {"1": 0, "2": 0, "3": 0, "4": 0}
        self.mask_counter = {"0": 0, "1": 0, "2": 0, "3": 0}
        self.social_distancing_counter = 0 # done
        self.splitting_level_counter = {"0": 0, "1": 0, "2": 0, "3": 0}

    def get_state(self):
        """Extracts the current state of the environment for the RL agent."""
        return [
            self.stats.count_health_status("healthy"),
            self.stats.count_health_status("infected"),
            self.stats.count_health_status("recovered"),
            self.stats.count_health_status("death"),
            self.stats.calculate_productivity(),
            self.current_step_in_day,
            int(self.social_distancing),
            int(self.mask_mandate),
        ]
        
    def initialize_agents(self):
        """Places agents in the grid and assigning them to sections. Starts the infection with a single 
        infected agent."""
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
        """Collects different data to be used to track performance of the model."""
        self.datacollector = DataCollector({
            "Healthy": lambda m: m.stats.count_health_status("healthy"),
            "Infected": lambda m: m.stats.count_health_status("infected"),
            "Recovered": lambda m: m.stats.count_health_status("recovered"),
            "Death": lambda m: m.stats.count_health_status("death"),
            "Productivity": lambda m: (m.stats.calculate_productivity() * 
                                     m.testing.get_productivity_modifier() * 
                                     m.grid_manager.get_cleaning_productivity_modifier()),
            #"Productivity": lambda m: m.get_shift_productivity_modifier(), #debugging
            "Quarantined": lambda m: len(m.quarantine.quarantine_zone),
            "Daily Infections": lambda m: m.stats.daily_infections,
            "Current Shift": lambda m: m.current_shift,
            "Shifts Per Day": lambda m: m.shifts_per_day
        })

    
    def should_change_shift(self):
        """Check if shift change should occur"""
        return self.current_step_in_day == self.next_shift_change

    def step(self, action=None):
        """Processes a single step in the model."""
        self.current_step += 1
        self.current_step_in_day = self.current_step % self.steps_per_day 

        if self.current_step_in_day == 0:
            self.current_day += 1

        self.process_scheduled_events()  # Runs all scheduled events for the current step

        pre_step_infected = self.stats.count_health_status("infected")
        self._process_agent_steps()  # Processes all agents' actions during this step
        post_step_infected = self.stats.count_health_status("infected")

        new_infections = max(0, post_step_infected - pre_step_infected)
        self.stats.update_infections(new_infections)  # Updates the infection count

        if self.current_step_in_day == self.steps_per_day - 1:
            self.stats.process_day_end()  # Gets daily stats

        self.datacollector.collect(self)
        self.testing.current_productivity_impact = 0  # Resets the testing impact on productivity

        # Return results for training and visualization
        return self._get_step_results(new_infections, post_step_infected)

        
    def _process_agent_steps(self):
        """Method to call each agent to get them to move in the environment for a step"""
        for agent in self.schedule.agents:
            if self.social_distancing and agent.pos is not None: #if social distancing is on call this function before step
                self.grid_manager.move_agent_social_distance(agent)
            agent.step() #step function in the agent class
    
    def process_scheduled_events(self):
        """Process all scheduled events in the correct order"""
        self.grid_manager.process_cleaning(self.current_step_in_day) #Call to process cleaning if correct day
        for testing_type in ['light', 'medium', 'heavy']: 
            if self.testing.should_run_testing(testing_type):
                self.testing.process_testing(testing_type) #If its a testing step, call the processing testing method in testing class
            
        self.quarantine.process_quarantine() #If an agent tests positive for the infection, throw them in quarantine, if they are ready to be taken out do that.
        
        if self.should_change_shift(): #Checks if we are on a shift change step.
            self.grid_manager.process_shift_change() #Processes the shift change in the grid manager class.
            self.next_shift_change = (self.current_step_in_day + self.steps_per_shift) % self.steps_per_day #Calculates the next shift change

    def get_steps_per_shift(self):
        """Helper function to get the steps per shift"""
        return self.steps_per_shift

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
        """Calculates the new step results after each step."""
        base_productivity = self.stats.calculate_productivity()
        cleaning_modifier = self.grid_manager.get_cleaning_productivity_modifier()
        testing_modifier = self.testing.get_productivity_modifier()

        final_productivity = (
            base_productivity * 
            cleaning_modifier * 
            testing_modifier
        )

        return {
            'day': self.current_day,
            'step_in_day': self.current_step_in_day,
            'new_infections': new_infections,
            'total_infected': total_infected,
            'productivity': final_productivity,
            'quarantined': len(self.quarantine.quarantine_zone),
            'base_production': base_productivity,
            'infection_penalty': -2.0 * (new_infections / self.num_agents),
            'cleaning_modifier': cleaning_modifier,
            'testing_modifier': testing_modifier
        }

    
    def update_config(self, action_dict):
        """Method to update the current factor health configuration. Allows for the 6 variables to be changed during a simulation"""
        if "cleaning_type" in action_dict:
            self.initial_cleaning = action_dict["cleaning_type"]
            self.grid_manager.set_cleaning_type(action_dict["cleaning_type"])
            self.cleaning_counter[action_dict["cleaning_type"]] += 1

        if "splitting_level" in action_dict:
            self.splitting_level = action_dict["splitting_level"]
            # self.splitting_level[action_dict["splitting_level"]] += 1

        if "testing_level" in action_dict:
            self.test_lvl = action_dict["testing_level"]
            self.testing.set_testing_level(action_dict["testing_level"])
            self.swab_testing_counter[action_dict["testing_level"]] += 1

        if "social_distancing" in action_dict:
            self.social_distancing = action_dict["social_distancing"]
            self.social_distancing_counter += 1

        if "mask_mandate" in action_dict:
            self.mask_mandate = action_dict["mask_mandate"]
            #self.mask_counter[str(action_dict["mask_mandate"])] += 1

        if "shifts_per_day" in action_dict:
            self.shifts_per_day = action_dict["shifts_per_day"]
            self.steps_per_shift = self.steps_per_day // self.shifts_per_day
            self.next_shift_change = (self.current_step_in_day + self.steps_per_shift) % self.steps_per_day
            self.grid_manager.process_shift_change()
            self.shifts_counter[str(action_dict["shifts_per_day"])] += 1