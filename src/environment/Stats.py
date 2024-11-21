class StatsCollector:
    """Class that collects data for the simulation"""
    def __init__(self, model):
        self.model = model
        self.current_day = 0
        self.daily_infections = 0
        self.temp_infections = 0
        self.daily_stats = []
        self.previous_productivity = None
        
    def count_health_status(self, status):
        """Counts how many healthy, infected, and recovered agents in the grid"""
        return sum(1 for agent in self.model.schedule.agents 
                  if agent.health_status == status)
                  
    def calculate_productivity(self):
        """Calculates the current productivity by summing each agents current_production value"""
        return sum(agent.current_production for agent in self.model.schedule.agents)
        
    def update_infections(self, new_infections):
        """Update infection counters"""
        self.temp_infections += new_infections
        
        if self.model.schedule.steps % self.model.steps_per_day == 0:
            self.daily_infections = self.temp_infections
            self.temp_infections = 0
        
    def process_day_end(self):
        """For processing daily stats. Not really useful in current implementation"""
        self.current_day += 1
        self.daily_stats.append({
            'day': self.current_day,
            'infections': self.daily_infections,
            'healthy': self.count_health_status("healthy"),
            'infected': self.count_health_status("infected"),
            'recovered': self.count_health_status("recovered"),
            'productivity': self.calculate_productivity()
        })
        
    def get_state(self):
        #Helper method to get the ccounts for health status.
        return [
            self.count_health_status("healthy"),
            self.count_health_status("infected"),
            self.count_health_status("recovered"),
        ]
        
    def is_done(self):
        """Checks if simulation is done"""
        return (self.count_health_status("infected") == 0 or 
                self.model.schedule.steps > 100)