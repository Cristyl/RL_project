import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import matplotlib.pyplot as plt

# Define the DQN agent class
class DQNAgent:
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.alpha = 0.1

    def _build_model(self):
        model = Sequential([
            InputLayer(shape=(self.observation_size,)),
            Dense(16, activation='relu'),
            Dense(self.action_size)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
        return model

    def remember(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))

    def act(self, observation):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(observation, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        memory_list = list(self.memory)
        # Randomly sample indices without replacement
        indices = np.random.choice(len(memory_list), batch_size, replace=False)
        # Extract the sampled experiences
        minibatch = [memory_list[i] for i in indices]

        # Prepare training data
        observations = np.zeros((batch_size, self.observation_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (observation, action, reward, next_observation, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = (1 - self.alpha) * reward + self.alpha * (reward + self.gamma * np.amax(self.model.predict(next_observation, verbose=0)[0]))
            
            # Compute the current Q value
            target_f = self.model.predict(observation, verbose=0)[0]
            target_f[action] = target
            observations[i] = observation
            targets[i] = target_f

        # Fit the model using the batch of observations and targets
        self.model.fit(observations, targets, batch_size=batch_size, epochs=1, verbose=0)

        # Decay the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the environment
env = gym.make('LunarLander-v2')
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(observation_size, action_size)

# Training loop
batch_size = 64
num_episodes = 100
episode_rewards = []
for episode in range(num_episodes):
    observation, info = env.reset()
    print("episode: ", episode)
    observation = np.reshape(observation, [1, observation_size])
    total_reward = 0
    for t in range(500):
        # Render the environment (optional)
        # env.render()

        # Choose an action
        action = agent.act(observation)

        # Perform the action
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = np.reshape(next_observation, [1, observation_size])

        # Remember the experience
        agent.remember(observation, action, reward, next_observation, terminated or truncated)

        # Update the state
        observation = next_observation

        total_reward += reward

        # Check if episode is finished
        if terminated or truncated:
            episode_rewards.append(total_reward)
            if total_reward >= 200:
                print("success with reward: ", total_reward)
            break

        # Train the agent
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()
