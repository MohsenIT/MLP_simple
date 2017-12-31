import numpy as np
from matplotlib import pyplot as plt, style as style

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
line_styles = ['-', '--', '-.', ':']


def plot_bipolar_scatter_and_discriminator_line(data_2d, target, learning_w, learning_names):
    plt.scatter(data_2d[target == -1, 0], data_2d[target == -1, 1], c='y', s=100, marker='_')
    plt.scatter(data_2d[target == 1, 0], data_2d[target == 1, 1], c='m', s=100, marker='+')

    for i in range(len(learning_w)):
        w = learning_w[i]
        x = np.arange(min(data_2d[:, 0]), max(data_2d[:, 0]) - min(data_2d[:, 0]) + 0.1, 0.1, float)
        y = (-w[0] / w[1]) * x + (-w[2] / w[1])
        plt.plot(x, y, linestyle=line_styles[i], color=colors[i], label=learning_names[i])

    style.use('ggplot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=3)
    plt.show()


def plot_bipolar_scatter_and_discriminator_circle(data_2d, target, w, legend_label):
    r = np.sqrt(-w[2] / w[0])
    circle = plt.Circle((0, 0), r, color=colors[0], label=legend_label, fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle)

    ax.scatter(data_2d[target == -1, 0], data_2d[target == -1, 1], c='y', s=100, marker='_')
    ax.scatter(data_2d[target == +1, 0], data_2d[target == +1, 1], c='m', s=100, marker='+')

    style.use('ggplot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=3)
    plt.show()


def plot_bipolar_scatter_and_discriminator_circle2(data_2d, target, w, legend_label):
    r = np.sqrt(7 / 18)
    circle = plt.Circle((7/6, 5/6), r, color=colors[0], label=legend_label, fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle)

    ax.scatter(data_2d[target == -1, 0], data_2d[target == -1, 1], c='y', s=100, marker='_')
    ax.scatter(data_2d[target == +1, 0], data_2d[target == +1, 1], c='m', s=100, marker='+')

    style.use('ggplot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=3)
    plt.show()


def plot_bipolar_scatter_and_discriminator_parallel_lines(data_2d, target, w, learning_name):
    plt.scatter(data_2d[target == -1, 0], data_2d[target == -1, 1], c='y', s=100, marker='_')
    plt.scatter(data_2d[target == 1, 0], data_2d[target == 1, 1], c='m', s=100, marker='+')

    x = np.arange(min(data_2d[:, 0]), max(data_2d[:, 0]) - min(data_2d[:, 0]) + 0.1, 0.1, float)
    y = np.sqrt(-w[1] / w[0]) * np.ones(x.shape[0])
    plt.plot(x,  y, linestyle=line_styles[0], color=colors[0], label=learning_name)
    plt.plot(x, -y, linestyle=line_styles[0], color=colors[0])

    style.use('ggplot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=3)
    plt.show()