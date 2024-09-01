import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

class Agent():
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.number_of_episodes = 100000
        self.number_of_steps = 500
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99999
        self.env = gym.make("CartPole-v1")
    
    def choose_action(self, state, epsilon):
        if random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            actions = [self.q_table.get((state, action), 0) for action in range(self.env.action_space.n)]
            return np.argmax(actions)
        
    def update_table(self, state, action, next_state, reward):
        actual_reward = self.q_table.get((state, action), 0)
        next_state_rewards = [self.q_table.get((next_state, action), 0) for action in range(self.env.action_space.n)]
        best_nsr = max(next_state_rewards)
        updated_reward = (1 - self.alpha) * actual_reward + self.alpha * (reward + self.gamma * best_nsr)
        self.q_table[(state, action)] = updated_reward

    def learn(self):
        rewards = []
        avg_rewards = []
        max_rewards = []
        min_rewards = []

        for episode in range(self.number_of_episodes):
            state, _ = self.env.reset()
            state = tuple(round(s, 1) for s in state)  # Discretize the state
            episode_reward = 0
            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            for _ in range(self.number_of_steps):
                action = self.choose_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                next_state = tuple(round(n, 1) for n in next_state)  # Discretize the next state
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
        episodes = np.arange(100, self.number_of_episodes + 1, 100)
        
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(episodes, avg_rewards, label='Average Reward')
        plt.plot(episodes, max_rewards, label='Max Reward')
        plt.plot(episodes, min_rewards, label='Min Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Performance over Episodes')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(rewards, label='Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode during Training')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return rewards

    def test(self, number_of_tests=100):
        test_rewards = []
        successful_episodes = 0
        original_epsilon = self.epsilon  # Store the original epsilon value
        self.epsilon = 0.0  # Disable exploration for testing

        for _ in range(number_of_tests):
            state, _ = self.env.reset()
            state = tuple(round(s, 1) for s in state)
            episode_reward = 0
            
            for _ in range(self.number_of_steps):
                action = self.choose_action(state, 0.0)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                next_state = tuple(round(n, 1) for n in next_state)
                episode_reward += reward

                state = next_state

                if terminated or truncated:
                    break
            
            test_rewards.append(episode_reward)
            
            if episode_reward >= 500:  # Check for successful episodes
                successful_episodes += 1
        
        self.epsilon = original_epsilon  # Restore the original epsilon value

        # Plotting the test results
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(test_rewards)
        plt.xlabel('Test Episode')
        plt.ylabel('Reward')
        plt.title('Rewards over Test Episodes')
        
        plt.subplot(1, 2, 2)
        plt.bar(['Successes', 'Failures'], [successful_episodes, number_of_tests - successful_episodes])
        plt.xlabel('Outcome')
        plt.ylabel('Number of Episodes')
        plt.title('Number of Successful Episodes (Reward = 500)')

        plt.tight_layout()
        plt.show()

        average_test_reward = np.mean(test_rewards)
        print(f"Average Test Reward: {average_test_reward}")
        print(f"Successful Episodes (Reward = 500): {successful_episodes}/{number_of_tests}")

# Create an agent instance
agent = Agent()
training_rewards = agent.learn()  # Train the agent

# Test the agent and plot results
agent.test(number_of_tests=100)
