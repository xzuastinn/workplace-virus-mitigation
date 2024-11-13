class TestingManager:
    def __init__(self, model):
        """Initialize the testing manager with specified rates for false results"""
        self.model = model
        self.false_positive_rate = 0.02  # 2% chance of false positive
        self.false_negative_rate = 0.15  # 15% chance of false negative
        self.tests_performed_today = 0
        self.last_test_day = -1

    def should_test_today(self):
        """Determine if testing should occur based on frequency"""
        current_day = self.model.schedule.steps // self.model.steps_per_day
        if current_day != self.last_test_day and current_day % self.model.test_frequency == 0:
            self.last_test_day = current_day
            self.tests_performed_today = 0
            return True
        return False

    def test_agent(self, agent):
        """
        Test an individual agent and return the test result.
        Accounts for false positives and false negatives.
        """
        import random

        is_actually_infected = agent.health_status == "infected"
        
        if is_actually_infected:
            return random.random() > self.false_negative_rate
        else:
            return random.random() < self.false_positive_rate

    def process_testing(self):
        """
        Process testing for all agents if it's a testing day. Quarantine agents who test positive.
        """
        if not self.should_test_today():
            return

        for agent in self.model.schedule.agents:
            if not agent.is_quarantined:
                test_positive = self.test_agent(agent)
                
                if test_positive:
                    self.model.quarantine.quarantine_agent(agent)
                    
                self.tests_performed_today += 1