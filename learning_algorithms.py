import numpy as np

from helper.plots import plot_bipolar_scatter_and_discriminator_line


def hopfield_learning(data):
    data_cnt = data.shape[0]
    dim = data.shape[1]
    w = np.zeros([dim, dim])
    for i in range(data_cnt):
        w += np.matmul(data[i].reshape(dim, 1), data[i].reshape(1, dim))
    np.fill_diagonal(w, 0)
    return w


def bam(s, t):
    w = np.zeros([s.shape[1], t.shape[1]])
    data_cnt = s.shape[0]
    for i in range(data_cnt):
        w += np.matmul(np.transpose(s), t)
    return w


def hebb_learning(data, target, w):
    for i in range(data.shape[0]):
        w += np.matmul(data[i].reshape(4, 1), target[i].reshape(1, 4))
    return w

def perceptron_learning(data, target, w, alpha=0.1, theta=0, epoch_cnt=10):
    for e in range(epoch_cnt):
        for i in range(target.size):
            y_in = np.dot(data[i], w)
            y = 1.0 if y_in > theta \
                else (-1.0 if y_in < -theta else 0)
            if target[i] != y:
                w += alpha * (data[i] * target[i])
                print(w)
    return w


def adaline_learning(data, target, w, alpha=0.1, epoch_cnt=10):
    for e in range(epoch_cnt):
        for i in range(target.size):
            y_in = np.dot(data[i], w)
            d_w = alpha * (target[i] - y_in) * data[i]
            # if np.max(np.abs(d_w)) < 0.005:  # termination condition
            #     return w
            w += d_w
            # # adaline debugging
            if i % 10 == 0:
                legend = 'adaline, i={:3d}, w=[{:04.4f}, {:04.4f}, {:04.4f}], d_w=[{:04.4f}, {:04.4f}, {:04.4f}]' \
                    .format(i, w[0], w[1], w[2], d_w[0], d_w[1], d_w[2])
                print(legend)
                plot_bipolar_scatter_and_discriminator_line(data, target, [w], [legend])
    return w


def adaline_learning_tanh(data, target, w, alpha=0.1, gamma_step_cnt=6, epoch_cnt=10):
    gamma = 0.001
    for l in range(gamma_step_cnt):
        for e in range(epoch_cnt):
            for i in range(target.size):
                y_in = np.dot(data[i], w)
                y = np.tanh(gamma * y_in)
                w += alpha * data[i] * (target[i] - y) * ((target[i] - y) ** 2)
        gamma *= 0.1
    return w

