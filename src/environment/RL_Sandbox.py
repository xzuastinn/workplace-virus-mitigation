from FactoryModel import factory_model
from src.environment.FactoryConfig import FactoryConfig

config = FactoryConfig.get_default_rl_config() #Gets the default starting rl configuration to run
model = factory_model(width=25, height=25, N=60, config=config) #how big the grid is with N agents with a given policy configs.

print("Initial Configuration:")
print(f"Cleaning Type: {model.initial_cleaning}")
print(f"Shifts Per Day: {model.shifts_per_day}")

# Run first half
for i in range(16):
    state, done, info = model.step()
    print(f"Step {i}: Productivity: {info['productivity']}, New Infections: {info['new_infections']}")

#Mid-simulation configuration update
action = {     
    'cleaning_type': 'medium',
    'splitting_level': 2,
    'testing_level': 'medium',
    'social_distancing': False,
    'mask_mandate': False,
    'shifts_per_day': 2
} 

print("\nUpdating Configuration Halfway:")
model.update_config(action)

print(f"New Cleaning Type: {model.initial_cleaning}")
print(f"New Shifts Per Day: {model.shifts_per_day}")

for i in range(16, 40):
    state, done, info = model.step()
    print(f"Step {i}: Productivity: {info['productivity']}, New Infections: {info['new_infections']}")
    if done:
        break