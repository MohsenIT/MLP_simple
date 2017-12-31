import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import learning_algorithms
from helper import data_preparation as func
from helper import data_digits as dt

'''DATA PREPARATION'''
data = dt.X

''''INITIALIZATION: weight init and shuffling data'''
noise_ratio = 0.5
in_data = np.copy(data.reshape([10, 64]))
in_data[np.random.rand(in_data.shape[0], in_data.shape[1]) < noise_ratio] = 0  # add noise
epoch_cnt = 100

'''a) LEARNING'''
w_hopfield = learning_algorithms.hopfield_learning(in_data)

'''b) MEMORIZE'''
indices = np.arange(0, in_data.shape[1])
print('Memorize Hopfield:')
for e in range(epoch_cnt):
    in_data = np.copy(data.reshape([10, 64]))
    in_data[np.random.rand(in_data.shape[0], in_data.shape[1]) < noise_ratio] = 0  # add noise

    # print("\n\n\n EPOCH %d IS STARTED\n" % e)
    for n in range(in_data.shape[0]):
        rnd.shuffle(indices)
        for i in indices:
            y_in_i = in_data[n, i] + np.matmul(in_data[n], w_hopfield[:, i])
            in_data[n, i] = np.sign(y_in_i)
        # print("\nDIGIT = %d" % n)
        # mem_digit = np.chararray((data.shape[1], data.shape[2]))
        # mem_digit[:] = ' '
        # mem_digit[in_data[n, :].reshape([data.shape[1], data.shape[2]]) == 1] = '#'
        # print(mem_digit.decode('utf-8'))

        # plt.imsave('img/%d.png' % n, -1 * np.copy(in_data[n, :].reshape([data.shape[1], data.shape[2]])), cmap=cm.gray)
        # plt.imsave('img/%do.png' % n, -1 * np.copy(   data[n, :].reshape([data.shape[1], data.shape[2]])), cmap=cm.gray)

    print(func.hamming_distance(in_data.reshape([data.shape[0], data.shape[1], data.shape[2]]), data))
