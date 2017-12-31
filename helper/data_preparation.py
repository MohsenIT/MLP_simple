import numpy as np


def generate_data(cnt, dim, low, high):
    data = np.random.rand(cnt, dim) * (high - low) + low
    target = y(data)
    return data, np.reshape(target, [cnt, 1])


def y(u):
    return (1 + u[:, 0] ** 0.5 + u[:, 1] ** (-1.0) + u[:, 2] ** (-1.5)) ** 2


def classification_accuracy(in_data, w, target):
    out = np.sign(np.matmul(in_data, w))
    return np.sum(np.int64(out * target) > 0) / in_data.shape[0]


def normalize_two_first_columns(in_data):
    x1 = 2 * (max(in_data[:, 0]) - in_data[:, 0]) / (max(in_data[:, 0]) - min(in_data[:, 0])) - 1
    x2 = 2 * (max(in_data[:, 1]) - in_data[:, 1]) / (max(in_data[:, 1]) - min(in_data[:, 1])) - 1
    return np.transpose(np.vstack([x1, x2, in_data[:, 2]]))


def normalize_four_first_columns(in_data):
    x1 = 2 * (max(in_data[:, 0]) - in_data[:, 0]) / (max(in_data[:, 0]) - min(in_data[:, 0])) - 1
    x2 = 2 * (max(in_data[:, 1]) - in_data[:, 1]) / (max(in_data[:, 1]) - min(in_data[:, 1])) - 1
    x3 = 2 * (max(in_data[:, 2]) - in_data[:, 2]) / (max(in_data[:, 2]) - min(in_data[:, 2])) - 1
    x4 = 2 * (max(in_data[:, 3]) - in_data[:, 3]) / (max(in_data[:, 3]) - min(in_data[:, 3])) - 1
    return np.transpose(np.vstack([x1, x2, x3, x4, in_data[:, 2]]))


def shuffle_data_and_target(data, target):
    all_data = np.hstack([data, target.reshape(target.size, 1)])
    np.random.shuffle(all_data)
    return all_data[:, 0:all_data.shape[1] - 1], all_data[:, all_data.shape[1] - 1]


def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return np.sum(s1 != s2)
