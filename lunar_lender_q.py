import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment without a time limit
env = gym.make("LunarLander-v2")

# Parameters for Q-learning
look_up_table = {}
gamma = 0.9
alpha = 0.1  # Fixed learning rate
num_episodes = 100000
epsilon = 1.0
epsilon_decay = 0.99995
epsilon_min = 0.01

# Lists to store data for plotting
rewards_per_episode = []
rewards_every_100 = []
epsilons = []
best_rewards = []
best_solution_r = 0  # Initialize the best reward

def select_action(state, epsilon):
    """Select an action using epsilon-greedy policy."""
    if random.random() <= epsilon:
        return env.action_space.sample()  # Exploration
    else:
        q_values = [look_up_table.get((state, action), 0) for action in range(env.action_space.n)]
        return np.argmax(q_values)

def update_q_table(state, action, reward, next_state):
    """Update the Q-table using the Q-learning formula."""
    current_q = look_up_table.get((state, action), 0)
    q_values_next = [look_up_table.get((next_state, a), 0) for a in range(env.action_space.n)]
    max_future_q = max(q_values_next)
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
    look_up_table[(state, action)] = new_q

for episode in range(num_episodes):
    observation, info = env.reset()
    observation = tuple(round(obs, 1) for obs in observation)
    total_reward = 0
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    # Update lists for plotting
    epsilons.append(epsilon)

    while True:
        action = select_action(observation, epsilon)
        observation1, reward, terminated, truncated, info = env.step(action)
        observation1 = tuple(round(obs, 1) for obs in observation1)
        total_reward += reward
        update_q_table(observation, action, reward, observation1)
        observation = observation1

        if terminated or truncated:
            if total_reward >= 200 and total_reward > best_solution_r:
                best_solution_r = total_reward
                best_rewards.append(best_solution_r)
            else:
                best_rewards.append(best_solution_r)
            break
    rewards_per_episode.append(total_reward)

    # Store rewards every 100 episodes for plotting
    if episode % 100 == 0:
        rewards_every_100.append(total_reward)
        print(f"Episode {episode}/{num_episodes}, Best Reward so far: {best_solution_r}")

print("Number of state-action pairs in lookup table:", len(look_up_table))
env.close()

# Testing phase using the learned policy (without exploration)
def test_policy(env, num_tests=100):
    test_rewards = []
    for _ in range(num_tests):
        observation, info = env.reset()
        observation = tuple(round(obs, 1) for obs in observation)
        total_reward = 0
        while True:
            q_values = [look_up_table.get((observation, action), 0) for action in range(env.action_space.n)]
            action = np.argmax(q_values)
            observation, reward, terminated, truncated, info = env.step(action)
            observation = tuple(round(obs, 1) for obs in observation)
            total_reward += reward
            if terminated or truncated:
                break
        test_rewards.append(total_reward)
    return test_rewards

# Run the final test
test_rewards = test_policy(env)
average_test_reward = np.mean(test_rewards)
print(f"Average reward over {len(test_rewards)} test episodes: {average_test_reward}")

# Plotting the data
plt.figure(figsize=(18, 12))

# Reward at every 100th Episode
plt.subplot(2, 2, 1)
plt.plot(range(0, num_episodes, 100), rewards_every_100, label='Reward every 100 Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward every 100 Episodes')
plt.grid(True)
plt.legend()

# Epsilon Over Time
plt.subplot(2, 2, 2)
plt.plot(epsilons, label='Epsilon')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay Over Time')
plt.grid(True)
plt.legend()

# Best Reward Over Time (every 100 episodes)
plt.subplot(2, 2, 3)
plt.plot(range(0, num_episodes, 100), best_rewards[::100], label='Best Reward every 100 Episodes')
plt.xlabel('Episode')
plt.ylabel('Best Reward')
plt.title('Best Reward Achieved Over Time (per 100 Episodes)')
plt.grid(True)
plt.legend()

# Test Rewards Plot
plt.subplot(2, 2, 4)
plt.plot(range(len(test_rewards)), test_rewards, label='Test Rewards')
plt.xlabel('Test Episode')
plt.ylabel('Reward')
plt.title('Rewards for Each Test Episode')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
