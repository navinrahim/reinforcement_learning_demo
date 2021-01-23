'''
https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
'''
import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# setup bins for output
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# dynamic setup of bin size based on environment high and low value
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE

# Initialize Q table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # Rewards from env is -1 until destination is reached where reward in 0

done = False

while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    env.render()

env.close()
