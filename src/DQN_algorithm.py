import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import time

# set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#define the DQN agent class
class DQNAgent:
    def __init__(self, seed = RANDOM_SEED):
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.lr = 0.001
        self.alpha = 0.1
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.number_of_episodes = 1000
        self.batch_size = 32
        self.update_target_freq = 4
        self.seed = seed

        # create the environment
        self.env = gym.make('LunarLander-v2')
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # set device to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _build_model(self):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(self.observation_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    #adds in memory the experience
    def remember(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))

    # chooses an action with epsilon greedy strategy
    def choose_action(self, observation):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
            q_values = self.model(observation)
            return torch.argmax(q_values).item()

    # replays some experiences
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        observations = []
        targets = []
        
        for index in minibatch:
            observation, action, reward, next_observation, done = self.memory[index]
            
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
            next_observation = torch.tensor(next_observation, dtype=torch.float32).to(self.device)
            
            current_q = self.model(observation).detach().clone()
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(self.target_model(next_observation)).item()
            
            current_q[action] = (1 - self.alpha) * current_q[action].item() + self.alpha * target
            observations.append(observation.unsqueeze(0))
            targets.append(current_q.unsqueeze(0))
        
        observations = torch.cat(observations)
        targets = torch.cat(targets)
        
        self.optimizer.zero_grad()
        predictions = self.model(observations)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

    def train(self):
        episode_rewards = []
        avg_rewards = []
        max_rewards = []
        min_rewards = []

        start_time = time.time()
        
        for episode in range(self.number_of_episodes):
            observation, _ = self.env.reset(seed=self.seed)
            self.seed += 1
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            total_reward = 0
            for _ in range(500):
                # choose an action
                action = self.choose_action(observation)
                # get the env info when executing the action
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                # remember the experience
                self.remember(observation, action, reward, next_observation, terminated or truncated)
                # set the state as the next one
                observation = next_observation

                if terminated or truncated:
                    break
                # replay experiences
                self.replay(self.batch_size)

            episode_rewards.append(total_reward)

            # update target network every few episodes
            if episode % self.update_target_freq == 0:
                self.update_target_model()
            
            # track plotting values
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                max_reward = np.max(episode_rewards[-50:])
                min_reward = np.min(episode_rewards[-50:])
                avg_rewards.append(avg_reward)
                max_rewards.append(max_reward)
                min_rewards.append(min_reward)
                print(f"Episode: {episode + 1}, Avg Reward: {avg_reward}, Max Reward: {max_reward}, Min Reward: {min_reward}")
            
        training_duration = time.time() - start_time
        print(f"Training completed in {training_duration:.2f} seconds.")
        
        return episode_rewards, avg_rewards, max_rewards, min_rewards, self.number_of_episodes
    
    def test(self, number_of_tests=100):
        successful_episodes = 0
        self.seed = 0
        test_rewards = []
        self.epsilon = 0.0
        for episode in range(number_of_tests):
            observation, _ = self.env.reset(seed=self.seed)
            self.seed += 1
            episode_reward = 0
            for _ in range(500):
                action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                observation = next_observation
                episode_reward += reward
                if terminated or truncated:
                    break
            if episode_reward >= 200:
                successful_episodes += 1
            print(f"Test Episode: {episode}, Reward: {episode_reward}")
            test_rewards.append(episode_reward)
        success_rate = 100.0 * successful_episodes / number_of_tests
        
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
        return success_rate

# initialize the DQN agent
agent = DQNAgent()

# train the agent
episode_rewards, avg_rewards, max_rewards, min_rewards, episodes = agent.train()

# plotting the average reward, max, and min rewards every 50 episodes
episodes_range = range(50, episodes + 1, 50)
plt.figure(figsize=(12, 6))
plt.plot(episodes_range, avg_rewards, label='Average Reward')
plt.plot(episodes_range, max_rewards, label='Max Reward')
plt.plot(episodes_range, min_rewards, label='Min Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Average, Max, and Min Reward per 50 Episodes')
plt.legend()
plt.show()

# test the trained agent
success_rate = agent.test()
print(f"Success Rate: {success_rate}%")
