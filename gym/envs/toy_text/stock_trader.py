import gym
import numpy as np
import sys
import pandas as pd
import random as pr

from gym import utils, spaces
from gym.utils import seeding

# data = pd.read_csv("/Users/seanlee/ReinforcementZeroToAll/TD.TO.csv", na_values=['null']).dropna(axis=0, how='any').reset_index()
data = pd.DataFrame([5,6,2,4,1], columns=['Adj Close'])

class StockTraderEnv(gym.Env):
    def __init__(self):
        self.index = 0
        self.transactions = []
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(5)
        self._seed()

        # Start the first game
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if self._stock_count() < 1 and action > 1:
            action = pr.randint(0, 1)
        reward = 0
        choice = action - 1
        actions = ['BUY', 'HOLD', 'SELL']
        row = data.loc[self.index]
        adj_close = float(row['Adj Close'])
        # print "%s %d" % (actions[action], adj_close)

        self.cash += choice * adj_close
        self.index = self.index + 1
        self.transactions.append([self.cash, choice])
        if len(self.transactions) >= data.shape[0]:
            done = True
            self.index = 0
        else:
            done = False

        # row = data.tail(1)
        # print float(row['Adj Close'])
        reward = (self.cash + self._stock_count() * float(row['Adj Close'])) - 10000

        # new_state = [self.cash, self._stock_count()]
        return self.index, reward, done, action

    def _reset(self):
        self.transactions, self.cash = [], 10000
        # cash, stock #, stock $
        return 0

    def _stock_count(self):
        count = 0
        for t in self.transactions:
            count += t[1]

        return count * -1