class QuarantineManager:
    """Class that handles how agents get sent to quarantine"""
    def __init__(self, model):
        self.model = model
        self.quarantine_zone = []
        self.quarantine_threshold = 1000 #Old functionality. Set this value to send any sick agents to quarentine after n steps of being sick.
        
    def process_quarantine(self):
        """Processes the quarantine process for sick agents."""
        for agent in self.model.schedule.agents:
            if (agent.health_status == "infected" and 
                agent.infection_time > self.quarantine_threshold and 
                not agent.is_quarantined): #Old functionality
                self.quarantine_agent(agent)

        for agent in self.quarantine_zone.copy(): #Checks if an infected agent is now recovered and can be returned to the grid.
            if agent.health_status == "recovered":
                self.return_from_quarantine(agent)
                
    def quarantine_agent(self, agent):
        """Send a sick or false positive agent into quarentine."""
        if agent.pos is None:
            return
        if agent not in self.quarantine_zone:
            if agent.pos is not None:
                agent.last_section = self.model.grid_manager.get_section_index(agent.pos[0]) #track the section they were in to be readded to
            self.model.grid.remove_agent(agent) #Pop them off the grid
            self.quarantine_zone.append(agent) #Add them to quarentine
            agent.is_quarantined = True
            
    def return_from_quarantine(self, agent):
        """Function to return a recovered agent from quarantine"""
        if agent in self.quarantine_zone:
            new_pos = self.model.grid_manager.get_valid_position(agent) #Gets position to drop off agent
            self.quarantine_zone.remove(agent)
            self.model.grid.place_agent(agent, new_pos)
            agent.is_quarantined = False
            agent.set_base_position(new_pos)
