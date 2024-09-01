import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

class TileCoder:
    def __init__(self, low, high, tiling_dims, num_tilings, offsets, bool_indices):
        self.num_tilings = num_tilings
        self.offsets = offsets
        self.low = low
        self.high = high
        self.tiling_dims = tiling_dims  # A list of tiling dimensions per state variable
        self.tile_widths = [(high[i] - low[i]) / (tiling_dims[i] - 1) if i not in bool_indices else 1 
                            for i in range(len(low))]
        self.bool_indices = bool_indices

    def get_tiles(self, state):
        tiles = []
        for i in range(self.num_tilings):
            offset = self.offsets[i]
            tile_indices = []
            for j in range(len(state)):
                if j in self.bool_indices:
                    # For boolean variables, directly use the state value (0 or 1)
                    tile_index = int(state[j])
                else:
                    # Calculate the tile index for continuous variables
                    tile_index = int((state[j] - self.low[j] + offset[j]) / self.tile_widths[j])
                tile_indices.append(tile_index)
            tiles.append(tuple(tile_indices))
        return tiles

class Agent():
    def __init__(self):
        self.alpha = 0.1 / 4
        self.gamma = 0.95
        self.number_of_episodes = 50000
        self.number_of_steps = 500
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99992
        self.env = gym.make("LunarLander-v2")
        
        # Define custom tiling dimensions for each state variable
        self.num_tilings = 4
        self.tiling_dims = [15, 15, 35, 35, 25, 35]  # Customize for continuous dimensions
        bool_indices = [6, 7]  # Assuming the last two dimensions are boolean
        
        # Random offsets for each tiling and each state variable
        self.offsets = [np.random.uniform(0, 0.2, size=self.env.observation_space.shape) for _ in range(self.num_tilings)]
        
        # Initialize TileCoder with customized tiling dimensions and boolean handling
        self.tile_coder = TileCoder(low=self.env.observation_space.low,
                                    high=self.env.observation_space.high,
                                    tiling_dims=self.tiling_dims,
                                    num_tilings=self.num_tilings,
                                    offsets=self.offsets,
                                    bool_indices=bool_indices)
        self.q_table = {}

    def choose_action(self, state, epsilon):
        tiles = self.tile_coder.get_tiles(state)
        if random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            actions = [sum(self.q_table.get((tiling, tile, action), 0) for tiling, tile in enumerate(tiles))
                       for action in range(self.env.action_space.n)]
            return np.argmax(actions)

    def update_table(self, state, action, next_state, reward):
        tiles = self.tile_coder.get_tiles(state)
        next_tiles = self.tile_coder.get_tiles(next_state)
        
        # Calculate the current Q-value estimate
        current_q = sum(self.q_table.get((tiling, tile, action), 0) for tiling, tile in enumerate(tiles))
        
        # Calculate the best Q-value for the next state
        next_qs = [sum(self.q_table.get((tiling, tile, next_action), 0) for tiling, tile in enumerate(next_tiles))
                   for next_action in range(self.env.action_space.n)]
        best_next_q = max(next_qs)
        
        # Update each tiling's Q-value
        for tiling, tile in enumerate(tiles):
            self.q_table[(tiling, tile, action)] = self.q_table.get((tiling, tile, action), 0) + self.alpha * (reward + self.gamma * best_next_q - current_q)

    def learn(self):
        rewards = []
        avg_rewards = []
        max_rewards = []
        min_rewards = []

        for episode in range(self.number_of_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            for _ in range(self.number_of_steps):
                action = self.choose_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward
                self.update_table(state, action, next_state, reward)

                state = next_state

                if terminated or truncated:
                    break
            rewards.append(episode_reward)

            # Every 100 episodes, calculate and store the average, max, and min rewards
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                max_reward = np.max(rewards[-100:])
                min_reward = np.min(rewards[-100:])
                avg_rewards.append(avg_reward)
                max_rewards.append(max_reward)
                min_rewards.append(min_reward)

                print(f"Episode: {episode+1}, Avg Reward (last 100): {avg_reward}, Max: {max_reward}, Min: {min_reward}, Epsilon: {self.epsilon}")

        # Plotting the results

        # Plot all rewards over episodes
        plt.figure(figsize=(12, 5))
        plt.plot(rewards, label='Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode during Training')
        plt.legend()
        plt.show()

        # Plot the average, max, and min rewards over time
        episodes = np.arange(100, self.number_of_episodes + 1, 100)
        plt.plot(episodes, avg_rewards, label='Average Reward')
        plt.plot(episodes, max_rewards, label='Max Reward')
        plt.plot(episodes, min_rewards, label='Min Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Performance over Episodes')
        plt.legend()
        plt.show()


    def test(self, number_of_tests=100):
        test_rewards = []
        original_epsilon = self.epsilon  # Store the original epsilon value
        self.epsilon = 0.0  # Disable exploration for testing

        for _ in range(number_of_tests):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.number_of_steps):
                action = self.choose_action(state, 0.0)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward

                state = next_state

                if terminated or truncated:
                    break
            
            test_rewards.append(episode_reward)
        
        self.epsilon = original_epsilon  # Restore the original epsilon value

        # Calculate the percentage of episodes with a reward > 200
        successful_episodes = sum(reward > 200 for reward in test_rewards)
        success_rate = (successful_episodes / number_of_tests) * 100

        # Plotting the test results
        plt.figure(figsize=(12, 5))
        plt.plot(test_rewards, label='Reward per Test Episode')
        plt.xlabel('Test Episode')
        plt.ylabel('Reward')
        plt.title('Rewards over Test Episodes')
        plt.legend()
        plt.show()

        average_test_reward = np.mean(test_rewards)
        print(f"Average Test Reward: {average_test_reward}")
        print(f"Percentage of Test Episodes with Reward > 200: {success_rate:.2f}%")

# Create an agent instance
agent = Agent()
agent.learn()  # Train the agent

# Test the agent and plot results
agent.test(number_of_tests=100)
