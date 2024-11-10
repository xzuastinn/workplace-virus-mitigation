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
        self.visualization = visualization  # Assign the visualization flag

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
            self.visualization_step()  # Call the simple step if in visualization mode
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
            return next_state, reward, done  # Return values only if not in visualization mode
      
    def visualization_step(self):
        """Advance the model one step without external actions (for visualization)."""
        reward = self.calculate_reward()
        self.current_reward += reward
        
        self.current_productivity = self.calculate_productivity()
        
        self.datacollector.collect(self)
        self.schedule.step()

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
        infection_reward = (self.count_health_status("healthy") / self.num_agents)
        productivity_penalty = -0.1 * (self.mask_mandate + self.social_distancing)
        vaccination_reward = 0.05 * self.num_vaccinated
        employee_reward = 0.1 * self.num_agents
        return infection_reward + productivity_penalty + vaccination_reward + employee_reward
    
    def calculate_productivity(self):
        """Calculate current productivity based on various factors"""
        base_productivity = 1.0
        
        # Productivity penalties
        if self.mask_mandate:
            base_productivity *= 0.95  # 5% reduction for mask mandate
        if self.social_distancing:
            base_productivity *= 0.90  # 10% reduction for social distancing
        
        #Productivity mainly based on the ratio of healthy agents to sick
        healthy_ratio = self.count_health_status("healthy") / self.num_agents
        base_productivity *= healthy_ratio  
        
        #Vaccines potentially cause productivity increase because workers not afraid of getting sick?
        vaccination_ratio = self.num_vaccinated / self.num_agents
        base_productivity *= (1 + 0.05 * vaccination_ratio)  #Small bonus up to 5% 
        
        return base_productivity
    
    def count_health_status(self, status):
        return sum(1 for agent in self.schedule.agents if agent.health_status == status)

    def is_done(self):
        return self.count_health_status("infected") == 0 or self.schedule.steps > 100
