import gymnasium as gym
import random
from gymnasium.wrappers import TimeLimit
env = gym.make("BipedalWalker-v3")
max_steps = 1600
env = TimeLimit(env, max_episode_steps=max_steps)
observation, info = env.reset()
look_up_table = {}
gamma = 0.9
epsylon = 0.3

#for certain number of times
for i in range(1000000):
    #choose action a (chooses randamly an action)
    random_num = random.uniform(0,1)
    r = 0
    visits = 0
    action = None
    observation = tuple(round(obs, 2) for obs in observation)
    if random_num >= epsylon:
        #choose best action so far
        for elem in look_up_table:
            if observation == elem[0]:
                if look_up_table[elem][0] > r:
                    action = elem[1]
                    r = look_up_table[elem][0]
    if random_num < epsylon or action is None:
        action = env.action_space.sample()
    action = tuple(round(act, 2) for act in action)
    #execute action a
    #observe bew state s'
    #collect reward r
    observation1, reward, terminated, truncated, info = env.step(action)
    observation1 = tuple(round(obs1, 2) for obs1 in observation1)
    #update or insert in table Q[s,a] as (1 - alpha)Q[s,a] + alpha(r + gamma X max(a')Q[s',a'])
    #define max_q as gretest reward for s'
    max_q = 0
    for elem in look_up_table:
        if observation1 == elem[0]:
            max_q = max(max_q, look_up_table[elem][0])
    visits = look_up_table.get((observation, action), (0, 0))[1]
    alpha = 1 / (1 + visits)
    reward = (1 - alpha) * reward + alpha * (reward + gamma * (max_q))
    if (observation, action) not in look_up_table:
        look_up_table[(observation, action)] = (reward, 0)
    else:
        #modify reward
        #modify reward in look_up_table
        current_reward, current_counter = look_up_table[(observation, action)]
        new_reward = reward
        new_counter = current_counter + 1
        look_up_table[(observation, action)] = (new_reward, new_counter)
    observation = observation1
    if terminated or truncated:
        observation, info = env.reset()
print("look_up_table_len: ", len(look_up_table))
env.close()