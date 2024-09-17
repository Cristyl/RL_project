import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class TileCoder:
    def __init__(self, low, high, tiling_dims, num_tilings, offsets, bool_indices):
        self.num_tilings = num_tilings
        self.offsets = offsets
        self.low = low
        self.high = high
        self.tiling_dims = tiling_dims
        self.tile_widths = [(high[i] - low[i]) / (tiling_dims[i] - 1) if i not in bool_indices else 1 
                            for i in range(len(low))]
        self.bool_indices = bool_indices

    def get_tiles(self, observation):
        tiles = []
        for i in range(self.num_tilings):
            offset = self.offsets[i]
            tile_indices = []
            for j in range(len(observation)):
                if j in self.bool_indices:
                    tile_index = int(observation[j])
                else:
                    tile_index = int((observation[j] - self.low[j] + offset[j]) / self.tile_widths[j])
                tile_indices.append(tile_index)
            tiles.append(tuple(tile_indices))
        return tiles

class Agent():
    def __init__(self, seed=RANDOM_SEED):
        self.alpha = 0.2 / 10
        self.gamma = 0.99
        self.number_of_episodes = 75000
        self.number_of_steps = 500
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.seed = seed
        
        # create the environment
        self.env = gym.make("LunarLander-v2")

        # setup values for TileCoder
        self.num_tilings = 10
        self.tiling_dims = [18, 18, 18, 18, 18, 18]
        bool_indices = [6, 7] 
        self.offsets = [np.random.uniform(0, 0.2, size=self.env.observation_space.shape) for _ in range(self.num_tilings)]
        self.tile_coder = TileCoder(low=self.env.observation_space.low,
                                    high=self.env.observation_space.high,
                                    tiling_dims=self.tiling_dims,
                                    num_tilings=self.num_tilings,
                                    offsets=self.offsets,
                                    bool_indices=bool_indices)
        # setup the q_table
        self.q_table = defaultdict(lambda:0.0)

    # choose an action with epsilon-greedy strategy
    def choose_action(self, observation):
        tiles = self.tile_coder.get_tiles(observation)
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            actions = [sum(self.q_table[(tiling, tile, action)] for tiling, tile in enumerate(tiles))
                       for action in range(self.env.action_space.n)]
            return np.argmax(actions)

    # update the Q-table
    def update_table(self, observation, action, next_observation, reward):
        tiles = self.tile_coder.get_tiles(observation)
        next_tiles = self.tile_coder.get_tiles(next_observation)
        
        current_q = sum(self.q_table[(tiling, tile, action)] for tiling, tile in enumerate(tiles))
        
        next_qs = [sum(self.q_table[(tiling, tile, next_action)] for tiling, tile in enumerate(next_tiles))
                   for next_action in range(self.env.action_space.n)]
        best_next_q = max(next_qs) if next_qs else 0
        
        for tiling, tile in enumerate(tiles):
            self.q_table[(tiling, tile, action)] += self.alpha * (reward + self.gamma * best_next_q - current_q)


    def learn(self):
        rewards = []
        avg_rewards = []
        max_rewards = []
        min_rewards = []
        epsilons = []
        epsilon_values = [1, 0.5, 0.1, 0.05, 0.01]
        epsilon_episodes = {eps: None for eps in epsilon_values}
        
        start_time = time.time()

        for episode in range(self.number_of_episodes):
            observation, _ = self.env.reset(seed=self.seed)
            self.seed += 1
            
            episode_reward = 0
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # capture the exact episode where epsilon crosses specific values
            for eps in epsilon_values:
                if np.isclose(self.epsilon, eps, atol=1e-4) and epsilon_episodes[eps] is None:
                    epsilon_episodes[eps] = episode + 1

            for _ in range(self.number_of_steps):
                # choose the action
                action = self.choose_action(observation)
                # get the env info when executing the action
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                # update the table
                self.update_table(observation, action, next_observation, reward)
                # set the observation as the next one
                observation = next_observation

                if terminated or truncated:
                    break

            rewards.append(episode_reward)

            # track plotting values
            if (episode + 1) % 500 == 0:
                avg_reward = np.mean(rewards[-500:])
                max_reward = np.max(rewards[-500:])
                min_reward = np.min(rewards[-500:])
                avg_rewards.append(avg_reward)
                max_rewards.append(max_reward)
                min_rewards.append(min_reward)
                epsilons.append(self.epsilon)

                print(f"Episode: {episode+1}, Avg Reward (last 500): {avg_reward}, Max: {max_reward}, Min: {min_reward}, Epsilon: {self.epsilon}")

        training_duration = time.time() - start_time
        print(f"Training completed in {training_duration:.2f} seconds.")

        episodes = np.arange(500, self.number_of_episodes + 1, 500)

        # plotting average, max, min rewards
        plt.figure(figsize=(12, 5))
        plt.plot(episodes, avg_rewards, label='Average Reward')
        plt.plot(episodes, max_rewards, label='Max Reward')
        plt.plot(episodes, min_rewards, label='Min Reward')
        plt.axhline(200, color='green', linestyle='--', label='Reward 200')

        # plot vertical lines for the exact episodes where epsilon changes
        for eps, episode in epsilon_episodes.items():
            if episode is not None:
                plt.axvline(episode, color='red', linestyle='--', label=f'Epsilon {eps:.2f} at {episode}')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Performance over Episodes')
        plt.legend()
        plt.show()

    def test(self, number_of_tests=100):
        test_rewards = []
        self.epsilon = 0.0

        for _ in range(number_of_tests):
            observation, _ = self.env.reset(seed=self.seed)
            self.seed += 1
            episode_reward = 0
            
            for _ in range(self.number_of_steps):
                action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                observation = next_observation

                if terminated or truncated:
                    break
            
            test_rewards.append(episode_reward)

        successful_episodes = sum(reward > 200 for reward in test_rewards)
        bad_episodes = sum(reward < 0 for reward in test_rewards)
        success_rate = (successful_episodes / number_of_tests) * 100
        failure_rate = (bad_episodes / number_of_tests) * 100

        average_test_reward = np.mean(test_rewards)
        print(f"Average Test Reward: {average_test_reward}")
        print(f"Percentage of Test Episodes with Reward > 200: {success_rate:.2f}%")
        print(f"Percentage of Test Episodes with Reward < 0: {failure_rate:.2f}%")

        # plotting the rewards from the test episodes
        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(1, number_of_tests + 1), test_rewards, label='Episode Reward')
        plt.axhline(200, color='green', linestyle='--', label='Reward 200')
        plt.axhline(0, color='red', linestyle='--', label='Reward 0')
        plt.xlabel('Test Episode')
        plt.ylabel('Reward')
        plt.title('Test Performance over Episodes')
        plt.legend()
        plt.show()

# initialize an agent
agent = Agent()

# train the agent
agent.learn()

# test the trained agent
agent.test()
