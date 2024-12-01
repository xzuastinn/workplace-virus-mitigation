class FactoryConfig:
    """Class that handles the configuration for the simulation. Has base
      visualization and base rl visualizations that can be changed"""
    def __init__(self, 
                 cleaning_type='light',
                 splitting_level=1,
                 testing_level='light',
                 social_distancing=False,
                 mask_mandate=2,
                 shifts_per_day=4,
                 steps_per_day=24,
                 width=25,
                 height=25,
                 num_agents=100,
                 visualization=False):
        
        self.cleaning_type = cleaning_type
        self.splitting_level = splitting_level
        self.testing_level = testing_level
        self.social_distancing = social_distancing
        self.mask_mandate = mask_mandate
        
        self.shifts_per_day = shifts_per_day
        self.steps_per_day = steps_per_day
        self.steps_per_shift = steps_per_day // shifts_per_day
        
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.visualization = visualization
        
    
    def update_from_action(self, action_dict):
        """
        Updates configuration based on RL action dictionary
        
        action_dict format:
        {
            'cleaning': (light, medium, heavy),
            'splitting': 0-3 (none, half, quarter, eighth),
            'testing': (none, light, medium, heavy),
            'social_distancing': (False, True),
            'mask_mandate': (False, True),
            'shifts': 0-3 (maps to 1, 2, 3, or 4 shifts per day)
        }
        """
        cleaning_map = ['light', 'medium', 'heavy']
        testing_map = ['none', 'light', 'medium', 'heavy']
        shifts_map = [1, 2, 3, 4]
        mask_map = [0,1,2,3]
        
        if 'cleaning' in action_dict:
            self.cleaning_type = cleaning_map[action_dict['cleaning']]
        if 'splitting' in action_dict:
            self.splitting_level = action_dict['splitting']
        if 'testing' in action_dict:
            self.testing_level = testing_map[action_dict['testing']]
        if 'social_distancing' in action_dict:
            self.social_distancing = bool(action_dict['social_distancing'])
        if 'mask_mandate' in action_dict:
            self.mask_mandate = mask_map[action_dict['mask_mandate']]
        if 'shifts' in action_dict:
            self.shifts_per_day = shifts_map[action_dict['shifts']]
            self.steps_per_shift = self.steps_per_day // self.shifts_per_day