import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from collections import deque
from random import randint
import os
import dqn
import random
import itertools
from os import path

df = pd.read_csv('AMZN.csv')

df2 = df.iloc[[0, -1]]

total_rows = df2.index.values[1]

start = pd.to_numeric(df2['Adj Close']).values.item(0)
end = pd.to_numeric(df2['Adj Close']).values.item(1)

def get_velocity(start, end, span) :
    return (end - start) / span

def get_v(data, span) :
    i = 0
    v = np.zeros(span).tolist()

    while i < len(data) - span :
        range = data[i:i+span]
        v.append(get_velocity(range.item(0), range.item(span - 1), span))
        i = i + 1

    return v

# velocity = get_velocity(start, end, total_rows)

# 0.0311424636872 => 3 cents a day
# print velocity
# print '---'

v5 = get_v(pd.to_numeric(df['Adj Close']).values, 5)
a5 = get_v(np.asarray(v5), 5)

dfV5 = pd.DataFrame(v5)
dfA5 = pd.DataFrame(a5)

v20 = get_v(pd.to_numeric(df['Adj Close']).values, 20)
a20 = get_v(np.asarray(v20), 20)

dfV20 = pd.DataFrame(v20)
dfA20 = pd.DataFrame(a20)

ma5 =  df['Adj Close'].rolling(window=5).mean()
ma20 =  df['Adj Close'].rolling(window=20).mean()

ema12 = df['Adj Close'].ewm(span=12).mean()
ema26 = df['Adj Close'].ewm(span=26).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9).mean()
oscillator = macd - signal

data = pd.concat([df.loc[:, ['Date', 'Adj Close']], ma5, ma20, dfA5, dfA20, macd, signal, oscillator], axis=1)
data.columns = ['Date', 'Adj Close', '5MA', '20MA', '5A', '20A', 'MACD', 'SIGNAL', 'OSCILLATOR']
# print pd.concat([df.loc[:, ['Date', 'Adj Close']], dfA5, dfA20], axis=1)

# print data.tail(360)

np_data = data.loc[data.shape[0] - 360: data.shape[0] - 356, 'Adj Close':].as_matrix()

# print np_data[4, 0]

# print Decimal(3 * (-19 * -5) / 100.0).quantize(Decimal('.1'), rounding=ROUND_FLOOR)
# print Decimal(3 * (-19 * -5) / 100.0).quantize(Decimal('.1'), rounding=ROUND_DOWN)
# print '{0}% {1} - {2}'.format((-19 * -5), round(3 * (-19 * -5) / 100.0, 2), round(round(3 * (-19 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-18 * -5), round(3 * (-18 * -5) / 100.0, 2), round(round(3 * (-18 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-17 * -5), round(3 * (-17 * -5) / 100.0, 2), round(round(3 * (-17 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-16 * -5), round(3 * (-16 * -5) / 100.0, 2), round(round(3 * (-16 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-15 * -5), round(3 * (-15 * -5) / 100.0, 2), round(round(3 * (-15 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-14 * -5), round(3 * (-14 * -5) / 100.0, 2), round(round(3 * (-14 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-11 * -5), round(3 * (-11 * -5) / 100.0, 2), round(round(3 * (-11 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-10 * -5), round(3 * (-10 * -5) / 100.0, 2), round(round(3 * (-10 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-9 * -5), round(3 * (-9 * -5) / 100.0, 2), round(round(3 * (-9 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-8 * -5), round(3 * (-8 * -5) / 100.0, 2), round(round(3 * (-8 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-7 * -5), round(3 * (-7 * -5) / 100.0, 2), round(round(3 * (-7 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-6 * -5), round(3 * (-6 * -5) / 100.0, 2), round(round(3 * (-6 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-5 * -5), round(3 * (-5 * -5) / 100.0, 2), round(round(3 * (-5 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-4 * -5), round(3 * (-4 * -5) / 100.0, 2), round(round(3 * (-4 * -5) / 100.0, 0)))
# print '{0}% {1} - {2}'.format((-3 * -5), round(3 * (-3 * -5) / 100.0, 2), round(round(3 * (-3 * -5) / 100.0, 0)))

# print '{0} {1} {2} {3}'.format(np_data[4, 0], 10000 // np_data[4, 0], math.floor((10000 // np_data[4, 0]) * (1 * 5 / 100.0)), 1 * 5)
# print '{0} {1} {2} {3}'.format(np_data[4, 0], 10000 // np_data[4, 0], math.floor((10000 // np_data[4, 0]) * (2 * 5 / 100.0)), 2 * 5)
# print '{0} {1} {2} {3}'.format(np_data[4, 0], 10000 // np_data[4, 0], math.floor((10000 // np_data[4, 0]) * (3 * 5 / 100.0)), 3 * 5)
# print '{0} {1} {2} {3}'.format(np_data[4, 0], 10000 // np_data[4, 0], math.floor((10000 // np_data[4, 0]) * (4 * 5 / 100.0)), 4 * 5)
# print '{0} {1} {2} {3}'.format(np_data[4, 0], 10000 // np_data[4, 0], math.floor((10000 // np_data[4, 0]) * (5 * 5 / 100.0)), 5 * 5)


# print(data.loc[data.shape[0] - 360: data.shape[0] - 356, 'Adj Close':].as_matrix())
# print(data.loc[data.shape[0] - 359: data.shape[0] - 355, 'Adj Close':].as_matrix())
#
# print(np.ravel(data.loc[data.shape[0] - 360: data.shape[0] - 356, 'Adj Close':].as_matrix())[20])
# print(np.ravel(data.loc[data.shape[0] - 359: data.shape[0] - 355, 'Adj Close':].as_matrix())[20])

# plt.figure()
# plt.plot(dfA5.tail(30))
# plt.plot(dfA20.tail(30))
# plt.grid()
# plt.show()

prop_count = 8
# take last 20 screens as input with 8 properties each (price, 5ma, 20ma, 5a, 20a, macd, signal, oscillator)
num_screen = 20
input_size = num_screen * prop_count + 1
output_size = 41
minibatch_size = 30

starting_point = 360

# discount factor
dis = 0.9
# buffer size
REPLAY_MEMORY = 50000

max_episodes = 2000
# store the previous observations in replay memory
replay_buffer = deque()

last_100_game_reward = deque()

csv = np.zeros((max_episodes, starting_point - num_screen))

def get_copy_var_ops(dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def _slice(data, start, span=19):
    return np.ravel(data.loc[data.shape[0] - start: data.shape[0] - (start - span), 'Adj Close':].as_matrix())

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            # a = argmax(mainDQN(s'))
            Q[0, action] = reward + dis * \
                                    targetDQN.predict(next_state)[
                                        0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

with tf.Session() as sess:
    mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
    targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
    tf.global_variables_initializer().run()

    # writer = tf.summary.FileWriter(path.abspath(path.join(os.sep, '/Users/seanlee', 'logdir')), sess.graph)

    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    sess.run(copy_ops)

    # e = 0.1

    for episode in range(max_episodes):
        e = 1. / ((episode / 10) + 1)
        # print("E: {}".format(e))
        done = False
        step_count = 1
        cash = 10000
        holdings = 0
        state = np.append(_slice(data, starting_point), 0)
        next_state = []
        rewards = []

        while not done:
            # del rewards[:]
            if np.random.rand(1) < e:
                # Explore
                # Random action between -20 and 20
                action = randint(-20, 20)
                # action = env.action_space.sample()
            else:
                # Exploit
                # Choose an action greedily from the Q-network
                action = np.argmax(mainDQN.predict(state)) - 20

            # Get new state and reward from environment
            # next_state, reward, done, _ = _step(action, step_count)
            next_state = _slice(data, starting_point - step_count)
            price = next_state[prop_count * num_screen - prop_count]

            if (action < 0 and holdings < 1) or action == 0:
                # cannot sell while no shares are held. thus do nothing
                reward = cash + (price * holdings)
                # print("{} \t{}% \tCash: {} \tHoldings: {} \tReward: {} \tsteps: {}".format(0, action*5, cash, holdings, reward, step_count))
            elif action < 0 and holdings > 0:
                qty = round(round(holdings * (action * -5) / 100.0, 0))
                holdings = holdings - qty
                cash = cash + (price * qty)
                reward = cash + (price * holdings)
                # print("{} \t{}% \tCash: {} \tHoldings: {} \tReward: {} \tsteps: {}".format(int(round(qty)), action*5, cash, holdings, reward, step_count))
            elif action > 0:
                qty = math.floor((cash // price) * (action * 5 / 100.0))
                holdings = holdings + qty
                cash = cash - (price * qty)
                reward = cash + (price * holdings)
                # print("{} \t{}% \tCash: {} \tHoldings: {} \tReward: {} \tsteps: {}".format(int(round(qty)), action*5, cash, holdings, reward, step_count))

            next_state = np.append(next_state, reward)

            if starting_point - step_count <= num_screen:
                done = True
            else:
                done = False

            csv[episode][step_count - 1] = reward
            # Save the experience to our buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            state = next_state
            step_count += 1

        # if e > 0.0001:
        #     e -= (0.1 - 0.0001) / max_episodes

        # print("Episode: {}  steps: {}".format(episode, step_count))

        if episode % 10 == 1:  # train every 10 episode
            # Get a random batch of experiences.
            for _ in range(50):
                minibatch_start = randint(0, len(replay_buffer) - minibatch_size)
                minibatch = list(itertools.islice(replay_buffer, minibatch_start, minibatch_start + minibatch_size))
                # minibatch = random.sample(replay_buffer, 25)
                loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)

        # copy q_net -> target_net
        sess.run(copy_ops)

        # print("Cash: {} Holdings: {} Price: {}".format(cash, holdings, next_state[20]))
        # last_100_game_reward.append(cash + (holdings * next_state[20]))
        #
        # if len(last_100_game_reward) > 100:
        #     last_100_game_reward.popleft()
        #
        #     avg_reward = np.mean(last_100_game_reward)
        #
        #     if avg_reward > 199:
        #         print("Game Cleared in {:f} episodes with avg reward {:f}".format(episode, avg_reward))
        #         break

    # pd.DataFrame(csv).to_csv('333.csv')

    # Predict on new state
    # print mainDQN.predict()

    plt.figure()
    plt.plot(csv[:, -1])
    plt.grid()
    plt.show()