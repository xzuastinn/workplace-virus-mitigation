class QuarantineManager:
    """Class that handles how agents get sent to quarantine"""
    def __init__(self, model):
        self.model = model
        self.quarantine_zone = []
        self.quarantine_duration = 40
        self.quarantine_timers = {}
        self.quarantine_threshold = 1000 #Old functionality. Set this value to send any sick agents to quarentine after n steps of being sick.
        
    def process_quarantine(self):
        """Processes the quarantine process for sick agents."""
        # Process new quarantines
        for agent in self.model.schedule.agents:
            if (agent.health_status == "infected" and 
                agent.infection_time > self.quarantine_threshold and 
                not agent.is_quarantined):
                self.quarantine_agent(agent)

        # Update quarantine timers and process releases
        for agent in list(self.quarantine_zone):  # Use list to avoid modification during iteration
            # Initialize timer if not exists
            if agent not in self.quarantine_timers:
                self.quarantine_timers[agent] = 0
                
            self.quarantine_timers[agent] += 1
            
            # Release conditions:
            # 1. Agent is recovered (original condition)
            # 2. Agent is healthy AND has been in quarantine for minimum duration
            # 3. Quarantine duration exceeded (catch-all for any status)
            if (agent.health_status == "recovered" or
                (agent.health_status == "healthy" and self.quarantine_timers[agent] >= self.quarantine_duration) or
                self.quarantine_timers[agent] >= self.quarantine_duration * 2):  # Max duration failsafe
                
                self.return_from_quarantine(agent)
                if agent in self.quarantine_timers:
                    del self.quarantine_timers[agent]
                
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
            self.quarantine_timers[agent] = 0 
            
    def return_from_quarantine(self, agent):
        """Function to return a recovered agent from quarantine"""
        if agent in self.quarantine_zone:
            attempts = 0
            max_attempts = 40
            valid_pos = None
            
            while attempts < max_attempts and valid_pos is None:
                pos = self.model.grid_manager.get_valid_position(agent)
                if (0 <= pos[0] < self.model.grid.width and 
                    0 <= pos[1] < self.model.grid.height):
                    valid_pos = pos
                attempts += 1
                
            if valid_pos is None:
                print(f"Warning: Could not find valid position for agent {agent.unique_id}")
                valid_pos = (0, 0)
                
            try:
                self.quarantine_zone.remove(agent)
                self.model.grid.place_agent(agent, valid_pos)
                agent.is_quarantined = False
                agent.set_base_position(valid_pos)
                if agent in self.quarantine_timers:
                    del self.quarantine_timers[agent]
            except Exception as e:
                print(f"Error placing agent {agent.unique_id} at position {valid_pos}: {str(e)}")
                if agent not in self.quarantine_zone:
                    self.quarantine_zone.append(agent)