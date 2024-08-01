import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

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
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.observation_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
        return model

    def remember(self, observation, action, reward, next_observation):
        self.memory.append((observation, action, reward, next_observation))

    def act(self, observation):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(observation)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for observation, action, reward, next_observation in minibatch:
            target = reward
            target = (1 - self.alpha) * reward + self.alpha * (reward + self.gamma * np.amax(self.model.predict(next_observation)[0]))
            target_f = self.model.predict(observation)
            target_f[0][action] = target
            self.model.fit(observation, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the environment
env = gym.make('LunarLander-v2', render_mode = "human")
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(observation_size, action_size)

# Training loop
batch_size = 32
num_episodes = 1000
for episode in range(num_episodes):
    observation, info = env.reset()
    obeservation = np.reshape(observation, [1, observation_size])
    for t in range(500):
        # Render the environment (optional)
        env.render()

        # Choose an action
        action = agent.act(observation)

        # Perform the action
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = np.reshape(next_observation, [1, observation_size])

        # Remember the experience
        agent.remember(observation, action, reward, next_observation)

        # Update the state
        observation = next_observation

        # Check if episode is finished
        if terminated or truncated:
            break

        # Train the agent
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)