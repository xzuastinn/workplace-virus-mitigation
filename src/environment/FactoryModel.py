import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from WorkerAgent import worker_agent

class factory_model(Model):
    """Factory environment where WorkerAgents interact."""
    def __init__(self, width, height, N, visualization=False):  # Add visualization as a parameter
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.central_line = width // 2
        self.mask_mandate = False
        self.social_distancing = False
        self.num_vaccinated = 0
        self.productivity = 1.0
        self.current_reward = 0.0
        self.visualization = visualization

        self.initialize_agents()
        self.datacollector = DataCollector(
            {
                "Healthy": lambda m: self.count_health_status("healthy"),
                "Infected": lambda m: self.count_health_status("infected"),
                "Recovered": lambda m: self.count_health_status("recovered"),
                "Productivity": lambda m: self.calculate_productivity(),
                "Total Reward": lambda m: self.current_reward
            }
        )

    def initialize_agents(self):
        first_infections = random.randrange(self.num_agents)
        for i in range(self.num_agents):
            side = 'left' if random.random() < 0.5 else 'right'
            worker = worker_agent(i, self, side)
            if i == first_infections:
                worker.health_status = "infected"
            self.schedule.add(worker)
            x = self.random.randrange(self.central_line) if side == 'left' else self.random.randrange(self.central_line, self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(worker, (x, y))

    def step(self, action=None):
        if self.visualization:
            self.visualization_step()
        else:
            if action is not None:
                self.apply_action(action)
            
            self.datacollector.collect(self)
            self.schedule.step()

            next_state = self.get_state()
            reward = self.calculate_reward()
            self.current_reward += reward
            self.current_productivity = self.calculate_productivity()

            done = self.is_done()
            return next_state, reward, done
      
    def visualization_step(self):
        """Advance the model one step without external actions (for visualization)."""
        self.datacollector.collect(self)
        self.schedule.step()
        reward = self.calculate_reward()
        self.current_reward += reward
        
        self.current_productivity = self.calculate_productivity()
        


    def apply_action(self, action):
        if action == 0:
            self.mask_mandate = not self.mask_mandate
        elif action == 1:
            self.social_distancing = not self.social_distancing
        elif action == 2:
            self.num_vaccinated = min(self.num_agents, self.num_vaccinated + 5)

    def get_state(self):
        healthy = self.count_health_status("healthy")
        infected = self.count_health_status("infected")
        recovered = self.count_health_status("recovered")
        return [healthy, infected, recovered, self.num_vaccinated]

    def calculate_reward(self):
        infection_reward = (self.count_health_status("healthy") + self.count_health_status("recovered") / self.num_agents)
        productivity_penalty = -0.1 * (self.mask_mandate + self.social_distancing)
        vaccination_reward = 0.05 * self.num_vaccinated
        employee_reward = 0.1 * self.num_agents
        return infection_reward + productivity_penalty + vaccination_reward + employee_reward
    
    def calculate_productivity(self):
        """Calculate current productivity based on various factors including recovered workers"""
        base_productivity = 1.0
        
        healthy_count = self.count_health_status("healthy")
        recovered_count = self.count_health_status("recovered")
        infected_count = self.count_health_status("infected")
        
        if self.mask_mandate:
            base_productivity *= 0.95  # 5% reduction for mask mandate
        if self.social_distancing:
            base_productivity *= 0.8  # 20% reduction for social distancing
        
        healthy_productivity = healthy_count
        recovered_productivity = recovered_count * 0.95
        infected_productivity = infected_count * 0.2
        
        total_effective_workforce = (healthy_productivity + recovered_productivity + infected_productivity) / self.num_agents
        base_productivity *= total_effective_workforce
        
        vaccination_ratio = self.num_vaccinated / self.num_agents
        base_productivity *= (1 + 0.05 * vaccination_ratio)
        
        return base_productivity
    
    def count_health_status(self, status):
        return sum(1 for agent in self.schedule.agents if agent.health_status == status)

    def is_done(self):
        return self.count_health_status("infected") == 0 or self.schedule.steps > 100
