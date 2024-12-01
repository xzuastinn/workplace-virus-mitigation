import random
class TestingManager:
    def __init__(self, model):
        self.model = model
        self.false_positive_rate = 0.05
        self.false_negative_rate = 0.14
        self.tests_performed = 0
        self.last_test_step = -1
        self.impact_duration_remaining = 0  # Track remaining impact steps
        self.current_test_impact = 0  
        
        
        self.testing_levels = { #Parameter dictionary for testing level. PRoportion is how many agents to test out of total pop
            'none': {
                'enabled': True,  # Default state
                'proportion': 0,
                'productivity_impact': 0,
                'frequency': 0,
                'impact_duration': 0
            },
            'light': {
                'enabled': False,
                'proportion': 0.2,
                'productivity_impact': 0.15, #How much of an impact this testing schedule has on productivity
                'frequency': 8, #How frequent the test schedule is ran
                'impact_duration': 4
            },
            'medium': {
                'enabled': False,
                'proportion': 0.4,
                'productivity_impact': 0.25,
                'frequency': 10,
                'impact_duration': 8
            },
            'heavy': {
                'enabled': False,
                'proportion': 0.65,
                'productivity_impact': 0.40,
                'frequency': 14,
                'impact_duration': 10
            }
        }
        
        self.next_test_steps = {
            level: 0
            for level, config in self.testing_levels.items()
        }
        

    def set_testing_level(self, level):
        """
        Enable specified testing level and disable others
        """
        if level not in ['none', 'light', 'medium', 'heavy']:
            raise ValueError("Invalid testing level")
            
        for test_level in self.testing_levels:
            self.testing_levels[test_level]['enabled'] = False
            
        if level != 'none':
            self.testing_levels[level]['enabled'] = True
            
    def test_agent(self, agent):
        """
        Test an individual agent and return the test result.
        Accounts for false positives and false negatives.
        """
        is_actually_infected = agent.health_status == "infected"
        
        if is_actually_infected:
            return random.random() > self.false_negative_rate
        else:
            return random.random() < self.false_positive_rate
            
    def get_agents_to_test(self, testing_intensity):
        """
        Select a proportion of agents to test based on testing intensity
        """
        all_agents = [agent for agent in self.model.schedule.agents 
                     if not agent.is_quarantined]
        num_agents_to_test = int(len(all_agents) * 
                               self.testing_levels[testing_intensity]['proportion'])
        return random.sample(all_agents, min(num_agents_to_test, len(all_agents)))
        
    def apply_productivity_impact(self, testing_intensity):
        """
        Apply productivity impact based on testing intensity
        """
        self.current_productivity_impact = self.testing_levels[testing_intensity]['productivity_impact']
        

    def should_run_testing(self, testing_type):
        """Helper method to compare the current step to when the next_test_step to determine if tests should be ran"""
        if not self.testing_levels[testing_type]['enabled']:
            return False
            
        if self.model.current_step_in_day == self.next_test_steps[testing_type]:
            self.next_test_steps[testing_type] = (
                self.model.current_step_in_day + 
                self.testing_levels[testing_type]['frequency']
            ) % self.model.steps_per_day #Resets next_test_steps to the next testing day
            return True #Test this step
        return False #Dont test this step
    
    def apply_testing_impact(self):
        """Apply the stored testing impact to all non-quarantined agents"""
        for agent in self.model.schedule.agents:
            if not agent.is_quarantined and not agent.is_dead:
                agent.being_tested = True
                agent.testing_impact = self.current_test_impact 
   
    def process_testing(self, testing_intensity):
        if not self.testing_levels[testing_intensity]['enabled']:
            return
                
        if self.should_run_testing(testing_intensity):
            print(f"Testing triggered: {testing_intensity} at step {self.model.current_step}")
            agents_to_test = self.get_agents_to_test(testing_intensity)
            
            self.last_test_step = self.model.current_step
            self.impact_duration_remaining = max(0, self.testing_levels[testing_intensity]['impact_duration'])
            self.current_test_impact = self.testing_levels[testing_intensity]['productivity_impact']

            for agent in agents_to_test:
                if not agent.is_dead:  # Add check for dead agents
                    test_positive = self.test_agent(agent)
                    if test_positive:
                        self.model.quarantine.quarantine_agent(agent)
                    self.tests_performed += 1
        
        # Always check impact duration and apply impact if needed
        if self.impact_duration_remaining > 0:
            self.apply_testing_impact()
            self.impact_duration_remaining -= 1

