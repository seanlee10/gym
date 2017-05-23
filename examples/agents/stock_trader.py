# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

register(
    id='StockTrader-v0',
    entry_point='gym.envs.toy_text:StockTraderEnv'
)

env = gym.make('StockTrader-v0')

# Initialize table with all zeros
Q = np.zeros([5, 3])
# Set learning parameters
dis = .99
num_episodes = 5

# create lists to contain total rewards and steps per episode
aList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    actions = []
    rewards = []

    # E greedy
    e = 1. / ((i // 100) + 1)  # Python2&3

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by e greedy
        # if np.random.rand(1) < e:
        #     action = env.action_space.sample()
        # else:
        #     action = np.argmax(Q[state, :])

        # action = env.action_space.sample()
        action = pr.randint(0, 2)

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        # Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state
        actions.append(_)
        rewards.append(reward)

    # print rewards
    # aList.append(int(reduce(lambda x, y: str(x)+str(y), actions), 3))
    # rList.append(rewards)
    # np.diff(rList).append(0)
    rewards = np.diff(rewards)
    # print actions
    for r, i in np.ndenumerate(rewards):
        print i, r
        # i = int(i)
        # Q[i, actions[i]] = r + dis * np.max(Q[i+1, :])
#
# arr_rewards = np.asarray(rList)
# # print np.max(arr_rewards[:,4])
# #
# print aList[np.argmax(arr_rewards[:,4])]
# print aList
# print np.diff(rList)
# print np.sum(rList)
# print np.column_stack((aList, arr_rewards[:,4]))

# for i, (r, a) in enumerate(np.column_stack((arr_rewards[:,4], aList))):
#     # print i
#     # print r
#     # print a
#     Q[i, a] = r + dis * np.max(Q[i+1])


# print("Success rate: " + str(sum(rList) / num_episodes))
# print("Final Q-Table Values")
print(Q)
# plt.bar(range(len(rList)), rList, color="blue")
# plt.show()