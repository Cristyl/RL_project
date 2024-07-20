import gymnasium as gym
import random
from gymnasium.wrappers import TimeLimit
env = gym.make("BipedalWalker-v3", render_mode = "human")
max_steps = 1600  # Specify the maximum number of steps
env = TimeLimit(env, max_episode_steps=max_steps)
observation, info = env.reset()
look_up_table = {}
alpha = 0.5
gamma = 0.9
epsylon = 0.3

#for certain number of times
for i in range(100000):
    #choose action a (chooses randamly an action)
    random_num = random.uniform(0,1)
    r = 0
    action = None
    observation = tuple(observation)
    if random_num >= epsylon:
        #choose best action so far
        for elem in look_up_table:
            if observation == elem[0]:
                if look_up_table[elem] > r:
                    action = elem[1]
                    r = look_up_table[elem]
    if random_num < epsylon or action is None:
        action = env.action_space.sample()
    action = tuple(action)
    #execute action a
    #observe bew state s'
    #collect reward r
    observation1, reward, terminated, truncated, info = env.step(action)
    observation1 = tuple(observation1)
    #update or insert in table Q[s,a] as (1 - alpha)Q[s,a] + alpha(r + gamma X max(a')Q[s',a'])
    #define max_q as gretest reward for s'
    max_q = 0
    elements = []
    for elem in look_up_table:
        if (observation1, action) == elem:
            max_q = max(max_q, look_up_table[elem])
    reward = (1 - alpha) * reward + alpha * (reward + gamma * (max_q))
    if (observation, action) not in look_up_table:
        look_up_table[(observation, action)] = reward
    else:
        #modify reward
        #modify reward in look_up_table
        look_up_table.update({(observation, action): reward})
    if terminated or truncated:
        observation, info = env.reset()
    if reward == 300:
        print("found policy")
        env.close()
print("look_up_table_len: ", len(look_up_table))
env.close()