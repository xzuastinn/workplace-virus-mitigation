class QuarantineManager:
    def __init__(self, model):
        self.model = model
        self.quarantine_zone = []
        self.quarantine_threshold = 1000
        
    def process_quarantine(self):
        for agent in self.model.schedule.agents:
            if (agent.health_status == "infected" and 
                agent.infection_time > self.quarantine_threshold and 
                not agent.is_quarantined):
                self.quarantine_agent(agent)

        for agent in self.quarantine_zone.copy():
            if agent.health_status == "recovered":
                self.return_from_quarantine(agent)
                
    def quarantine_agent(self, agent):
        if agent not in self.quarantine_zone:
            if agent.pos is not None:
                agent.last_section = self.model.grid_manager.get_section_index(agent.pos[0])
            self.model.grid.remove_agent(agent)
            self.quarantine_zone.append(agent)
            agent.is_quarantined = True
            
    def return_from_quarantine(self, agent):
        if agent in self.quarantine_zone:
            new_pos = self.model.grid_manager.get_valid_position(agent)
            self.quarantine_zone.remove(agent)
            self.model.grid.place_agent(agent, new_pos)
            agent.is_quarantined = False
            agent.set_base_position(new_pos)
