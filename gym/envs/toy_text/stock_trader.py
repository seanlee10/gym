import gym
import numpy as np
import sys
import pandas as pd
import random as pr

from gym import utils, spaces
from gym.utils import seeding

# data = pd.read_csv("/Users/seanlee/gym/gym/envs/toy_text/TD.TO.csv", na_values=['null']).dropna(axis=0, how='any').reset_index()
data = pd.DataFrame([5,6,7,8,9], columns=['Adj Close'])
# data = pd.DataFrame([5,6,2,4,1], columns=['Adj Close'])

class StockTraderEnv(gym.Env):
    def __init__(self):
        self.index = 0
        self.transactions = []
        self.action_space = spaces.Discrete(41)
        self.observation_space = spaces.Discrete(data.shape[0])
        self._seed()

        # Start the first game
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        action = action - 20
        if self._stock_count() < 1 and action > 1:
            action = pr.randint(-20, 0)

        row = data.loc[self.index]
        adj_close = float(row['Adj Close'])

        quotient = self.cash // adj_close

        if action > 0:
            shares = ((self._stock_count() * action * 5.) // 100)
        else:
            shares = ((quotient * action * 5.) // 100.)

        # print self.cash
        # print "%d %d%% %d x %d" % (action, action * 5, shares, adj_close)

        self.cash += shares * adj_close

        # print "cash: %.2f, stock: %.2f(%d)" % (self.cash, shares * adj_close, shares)

        #self.cash += choice * adj_close
        self.index = self.index + 1
        self.transactions.append([self.cash, shares])
        if len(self.transactions) >= data.shape[0]:
            done = True
            self.index = 0
        else:
            done = False

        # row = data.tail(1)
        # print float(row['Adj Close'])
        reward = ((self.cash + self._stock_count() * adj_close) - 10000)/10000
        # print "count: %d, reward: %.2f" % (self._stock_count(), reward)
        # print "----------"
        # new_state = [self.cash, self._stock_count()]
        return self.index, (1 / (1 +np.exp(-reward))), done, action

    def _reset(self):
        self.transactions, self.cash = [], 10000
        # cash, stock #, stock $
        return 0

    def _stock_count(self):
        count = 0
        for t in self.transactions:
            count += t[1]

        return count * -1