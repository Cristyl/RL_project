import math

epsilon_initial = 1.0
epsilon_target = 0.1
epsilon_decay = 0.99992

# Calculate the number of episodes required for epsilon to reach the target
t = math.log(epsilon_target / epsilon_initial) / math.log(epsilon_decay)
print(t)
