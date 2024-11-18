import random
class TestingManager:
    def __init__(self, model):
        self.model = model
        self.false_positive_rate = 0.02
        self.false_negative_rate = 0.15
        self.tests_performed = 0
        self.last_test_step = -1
        
        self.testing_levels = {
            'light': {
                'enabled': False,
                'proportion': 0.2,
                'productivity_impact': 0.15,
                'frequency': 8
            },
            'medium': {
                'enabled': False,
                'proportion': 0.5,
                'productivity_impact': 0.25,
                'frequency': 16
            },
            'heavy': {
                'enabled': False,
                'proportion': 0.8,
                'productivity_impact': 0.40,
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
        if not self.testing_levels[testing_type]['enabled']:
            return False
            
        if self.model.current_step_in_day == self.next_test_steps[testing_type]:
            self.next_test_steps[testing_type] = (
                self.model.current_step_in_day + 
                self.testing_levels[testing_type]['frequency']
            ) % self.model.steps_per_day
            return True
        return False
        
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

        agents_to_test = self.get_agents_to_test(testing_intensity)
        self.apply_productivity_impact(testing_intensity)
        print(f"Testing triggered: {testing_intensity} at step {current_step}")

        for agent in agents_to_test:
            test_positive = self.test_agent(agent)
            if test_positive:
                self.model.quarantine.quarantine_agent(agent)
            self.tests_performed += 1
