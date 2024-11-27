import random
class TestingManager:
    def __init__(self, model):
        self.model = model
        self.false_positive_rate = 0.05
        self.false_negative_rate = 0.14
        self.tests_performed = 0
        self.last_test_step = -1
        
        self.testing_levels = { #Parameter dictionary for testing level. PRoportion is how many agents to test out of total pop
            'light': {
                'enabled': False,
                'proportion': 0,
                'productivity_impact': 0, #How much of an impact this testing schedule has on productivity
                'frequency': 8 #How frequent the test schedule is ran
            },
            'medium': {
                'enabled': False,
                'proportion': 0.5,
                'productivity_impact': 0.10,
                'frequency': 16
            },
            'heavy': {
                'enabled': False,
                'proportion': 0.1,
                'productivity_impact': 0.20,
                'frequency': 24
            }
        }
        
        self.next_test_steps = {
            level: config['frequency'] 
            for level, config in self.testing_levels.items()
        }
        
        self.current_productivity_impact = 0

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
        
    def get_productivity_modifier(self):
        """
        Return the current productivity modifier from testing
        """
        return 1 - self.current_productivity_impact
        

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
        
    def process_testing(self, testing_intensity):
        """
        Process testing for selected agents based on testing intensity.
        Quarantine agents who test positive.
        """
        if not self.testing_levels[testing_intensity]['enabled']:
            return
                
        current_step = self.model.current_step
        
        if current_step == self.last_test_step:
            return
                
        self.last_test_step = current_step

        agents_to_test = self.get_agents_to_test(testing_intensity) #Agents to test by the proportion of tests conducted on population.
        self.apply_productivity_impact(testing_intensity) #Applies the productivity impact on the environment during a testing step
        print(f"Testing triggered: {testing_intensity} at step {current_step}")

        for agent in agents_to_test: #Runs the test and sends positive result agents to quarantine
            test_positive = self.test_agent(agent)
            if test_positive:
                self.model.quarantine.quarantine_agent(agent)
            self.tests_performed += 1
