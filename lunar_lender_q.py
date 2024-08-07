import gymnasium as gym
import random
import matplotlib.pyplot as plt

# Initialize the environment without a time limit
env = gym.make("LunarLander-v2")

# Parameters for Q-learning
look_up_table = {}
gamma = 0.8
alpha = 0.1  # Fixed learning rate
num_episodes = 1000
epsilon = 0.99
epsilon_decay = 0.95

# List to store total rewards per episode
rewards_per_episode = []

def select_action(state, epsilon):
    """Select an action using epsilon-greedy policy."""
    if random.random() < epsilon:
        # Exploration: choose a random action from a discrete space
        return env.action_space.sample()
    else:
        # Exploitation: choose the best-known action based on state
        q_values = {a[1]: look_up_table.get(a, 0) for a in look_up_table if a[0] == state}
        
        if not q_values:
            return env.action_space.sample()  # Default to a random action if no known actions
        
        # Return the action with the highest Q-value
        return max(q_values, key=q_values.get)

def update_q_table(state, action, reward, next_state):
    """Update the Q-table using the Q-learning formula."""
    current_q = look_up_table.get((state, action), 0)
    
    # Find the maximum Q-value for the next state
    q_values = {a[1]: look_up_table.get(a, 0) for a in look_up_table if a[0] == next_state}
        
    if not q_values:
        max_future_q = 0
    else:
        max_future_q = max(q_values, key=q_values.get)
    # Q-learning update
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
    look_up_table[(state, action)] = new_q

for episode in range(num_episodes):
    observation, info = env.reset()
    observation = tuple(round(obs, 1) for obs in observation)
    action_list = []
    total_reward = 0
    best_solution_r = 0
    epsilon = epsilon * epsilon_decay

    while True:  # Run until the episode naturally ends
        # Select action
        action = select_action(observation, epsilon)
        
        # Round each element of the action tuple
        action_list.append(action)

        # Take action and observe new state and reward
        observation1, reward, terminated, truncated, info = env.step(action)
       
        observation1 = tuple(round(obs, 1) for obs in observation1)
        total_reward += reward

        # Update Q-table
        update_q_table(observation, action, reward, observation1)

        # Transition to the next state
        observation = observation1

        # Check if the episode is finished
        if terminated or truncated:
            if total_reward >= 200 and total_reward > best_solution_r:
                best_solution_r = total_reward
                print("Action List:", action_list)
                print("Best Reward:", best_solution_r)
                print("Episode:", episode)
            break

    # Append total reward to the list
    rewards_per_episode.append(total_reward)

print("Number of state-action pairs in lookup table:", len(look_up_table))
env.close()

# Plotting the rewards over episodes
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.legend()
plt.show()
