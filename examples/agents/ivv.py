import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
from random import randint
import os
import random
import itertools
from os import path
from sklearn.model_selection import train_test_split

df = pd.read_csv('IVV.csv')

close_list = df['Adj Close'].values
total_rows = close_list.shape[0]
window_size = 25

IMG_DIR = os.path.dirname(os.path.abspath(__file__)) + '/images/'

i = 0
# plot = np.array(total_rows, 2)

csv = pd.DataFrame()

print(total_rows)
print('----------')

while i < total_rows - window_size - 1:
    window = close_list[i:i+window_size]
    min = window.min()
    max = window.max()

    def normalize(x):
        # print(min, max, x)
        return (x - min) / (max - min)

    label = (close_list[i+window_size+1] / close_list[i+window_size]) - 1
    label = round(label * 100, 2)

    if label > 1:
        label = 1
    elif label < -1:
        label = -1
    else:
        label = 0

    row = pd.Series([str(i) + '.jpg', label])
    csv = csv.append(row, ignore_index=True)

    #

    # print(list(map(normalize, window)))

    line_plot = pd.DataFrame(list(map(normalize, window)))

    line_plot.plot(kind='line', legend=False)
    plt.axis('off')
    # plt.show()
    plt.savefig(IMG_DIR + str(i) + '.jpg', bbox_inches='tight', pad_inches=0)

    i += 1

train, test = train_test_split(csv, test_size=0.25, shuffle=True)

train.to_csv('train.csv', header=False, index=False)
test.to_csv('test.csv', header=False, index=False)
# plt.axis('off')
# plt.show()
# plt.savefig(IMG_DIR + '0.png', bbox_inches='tight', pad_inches=0)

df2 = df.iloc[[0, -1]]
# print('2')
# print(df2.tail(5))

# total_rows = df2.index.values[1]

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

# print(data.tail(360))

# np_data = data.loc[data.shape[0] - 360: data.shape[0] - 356, 'Adj Close':].as_matrix()

# print(np_data[4, 0])

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
