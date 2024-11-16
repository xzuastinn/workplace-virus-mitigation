import random
class TestingManager:
    def __init__(self, model):
        """Initialize the testing manager with specified rates for false results"""
        self.model = model
        self.false_positive_rate = 0.02  # 2% chance of false positive
        self.false_negative_rate = 0.15  # 15% chance of false negative
        self.tests_performed = 0
        self.last_test_step = -1

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

    def process_testing(self):
        """
        Process testing for all agents. Quarantine agents who test positive.
        """
        current_step = self.model.schedule.steps
        
        if current_step == self.last_test_step:
            return
            
        self.last_test_step = current_step
        print(f"Processing tests at step {current_step}")
        
        for agent in self.model.schedule.agents:
            if not agent.is_quarantined:
                test_positive = self.test_agent(agent)
                
                if test_positive:
                    self.model.quarantine.quarantine_agent(agent)
                    
                self.tests_performed += 1