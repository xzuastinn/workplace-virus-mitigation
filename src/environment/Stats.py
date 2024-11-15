class StatsCollector:
    def __init__(self, model):
        self.model = model
        self.current_day = 0
        self.daily_infections = 0
        self.temp_infections = 0
        self.daily_stats = []
        self.previous_productivity = None
        
    def count_health_status(self, status):
        return sum(1 for agent in self.model.schedule.agents 
                  if agent.health_status == status)
                  
    def calculate_productivity(self):
        return sum(agent.current_production for agent in self.model.schedule.agents)
        
    def update_infections(self, new_infections):
        """Update infection counters"""
        self.temp_infections += new_infections
        
        if self.model.schedule.steps % self.model.steps_per_day == 0:
            self.daily_infections = self.temp_infections
            self.temp_infections = 0
        
    def process_day_end(self):
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
        return [
            self.count_health_status("healthy"),
            self.count_health_status("infected"),
            self.count_health_status("recovered"),
            self.model.num_vaccinated
        ]
        
    def is_done(self):
        return (self.count_health_status("infected") == 0 or 
                self.model.schedule.steps > 100)