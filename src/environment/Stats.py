class StatsCollector:
    def __init__(self, model):
        self.model = model
        self.current_reward = 0.0
        self.cumulative_reward = 0.0
        self.current_day = 0
        self.daily_infections = 0
        self.daily_stats = []
        
    def count_health_status(self, status):
        return sum(1 for agent in self.model.schedule.agents 
                  if agent.health_status == status)
                  
    def calculate_productivity(self):
        return sum(agent.current_production for agent in self.model.schedule.agents)
        
    def calculate_reward(self, new_infections):
        productivity = self.calculate_productivity()
        base_reward = productivity * 2.0
        infection_penalty = -2.0 * (new_infections / self.model.num_agents)
        return base_reward + infection_penalty
        
    def update_infections(self, new_infections):
        self.daily_infections += new_infections
        
    def update_reward(self, reward):
        self.current_reward = reward
        self.cumulative_reward += reward
        
    def process_day_end(self):
        self.current_day += 1
        self.daily_stats.append({
            'day': self.current_day,
            'infections': self.daily_infections,
            'healthy': self.count_health_status("healthy"),
            'infected': self.count_health_status("infected"),
            'recovered': self.count_health_status("recovered"),
            'productivity': self.calculate_productivity(),
            'cumulative_reward': self.cumulative_reward
        })
        self.daily_infections = 0
        
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