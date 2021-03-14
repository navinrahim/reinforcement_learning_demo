'''
https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
'''
import gym
import numpy as np
import time

LEARNING_RATE = 0.1  # can be modified, can be a value between 0 and 1
DISCOUNT = 0.95  # How important are future rewards over current rewards
EPISODES = 2000

DISPLAY_EPISODE = 500  # Display metrics after how many episodes

epsilon = 0.5 # amount of randomness/exploration, the higher epsilon -> more randomness
START_EPSILON_DECAYING_EPISODE = 1
END_EPSILON_DECAYING_EPISODE = EPISODES // 2
epsilon_decay_factor = epsilon / (END_EPSILON_DECAYING_EPISODE - START_EPSILON_DECAYING_EPISODE)

env = gym.make("MountainCar-v0")
# env.reset() # reset the env and return the initial state

# setup bins for output
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

# dynamic setup of bin size based on environment high and low value
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE

# Initialize Q table
# Rewards from env is -1 until destination is reached where reward in 0
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# analysis variables
episode_rewards = []
aggregate_episode_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

# Convert state from env to bins. The output can be used to query the qtable
# Eg: q_table[get_discrete_state(env.reset())]


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))




# Run for multiple episodes(iterations)
for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())  # get initial state

    if episode % DISPLAY_EPISODE == 0:
        print(episode)
        render = True
    else:
        render = False

    done = False
    # action 0-> go back; 1-> no action; 2->go forward
    while not done:
        if(np.random.random() > epsilon):
            # perform action based on value of Q table
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            # the Q-value that would be occured with current Q-table with new state
            max_future_q = np.max(q_table[new_discrete_state])
            # pick the current Q-table value
            current_q = q_table[discrete_state][action]
            # Q-Learning formula
            new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # update new Q value. Current state's Q table value is based on the output state
            q_table[discrete_state][action] = new_q
        # if new position is the end goal, then update Q_value with highest reward
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state][action] = 0

        discrete_state = new_discrete_state

    if START_EPSILON_DECAYING_EPISODE <= episode <= END_EPSILON_DECAYING_EPISODE:
        epsilon -= epsilon_decay_factor

    episode_rewards.append(episode_reward)
    
env.close()
