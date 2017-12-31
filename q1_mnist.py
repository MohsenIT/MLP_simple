import numpy as np

import helper.data_preparation as dt
import helper.mnist_loader as mnist

'''DATA PREPARATION'''
mnist_train, _, mnist_test = mnist.load_data()
train_data, train_target = mnist_train[0], mnist.vectorize_target(mnist_train[1])
test_data, test_target = mnist_test[0], mnist_test[1]

''''INITIALIZATION: weight init and shuffling data'''
train_size = train_data.shape[0]  # training set size
nn_input_dim = train_data.shape[1]  # input layer dimensionality
nn_output_dim = 10  # output layer dimensionality

# Gradient descent parameters
learning_rate = 0.01  # learning rate for gradient descent
reg_lambda = 0.0000  # regularization strength

x, y = train_data, train_target


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


# feed forward of a batch
def predict(weights, batch):
    w1, b1, w2, b2 = weights['w1'], weights['b1'], weights['w2'], weights['b2']
    z1 = np.dot(batch, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


# evaluate the total loss between target and output considering weights norm
def calculate_loss(output, target, weights):
    w1, w2 = weights['w1'], weights['w2']
    target_loss_term = np.mean(np.square(output - target))
    weight_regularization_term = reg_lambda * 0.5 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return target_loss_term + weight_regularization_term


# this function learns parameters for the neural network and returns the model.
def build_model(hidden_nodes_cnt, num_passes=2000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    w1 = np.random.randn(nn_input_dim, hidden_nodes_cnt) / 100
    b1 = np.random.randn(hidden_nodes_cnt) / 100
    w2 = np.random.randn(hidden_nodes_cnt, nn_output_dim) / 100
    b2 = np.random.randn(nn_output_dim) / 100
    current_model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        # call
        z1, a1, z2, a2 = predict(current_model, x)
        output = a2
        if print_loss and i % 2 == 0:
            if i == 0:
                print("iteration\ttrain acc\ttest acc\tloss")
            train_accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
            test_output = predict(current_model, test_data)[3]
            test_accuracy = np.mean(np.argmax(test_output, axis=1) == test_target)
            loss = calculate_loss(output, y, current_model)
            print("%i\t%f\t%f\t%f" % (i + 1, train_accuracy, test_accuracy, loss))

        # recall
        derv_out2 = (y - a2) * (a2 * (1 - a2))
        delta2 = np.matmul(np.transpose(a1), derv_out2) / train_size
        dw2 = delta2 + reg_lambda * w2
        db2 = np.mean(b2 * derv_out2, 0)

        derv_out1 = a1 * np.reshape(np.sum(delta2 * w2, 1), [1, a1.shape[1]])
        delta1 = np.matmul(np.transpose(x), derv_out1) / train_size
        dw1 = delta1 + reg_lambda * w1
        db1 = np.mean(b1 * derv_out1, 0)

        # gradient descent parameter update
        w1 += learning_rate * dw1
        b1 += learning_rate * db1
        w2 += learning_rate * dw2
        b2 += learning_rate * db2

        # assign new parameters to the model
        current_model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return current_model


'''a) LEARNING'''
# Build a model with a 3-dimensional hidden layer
model = build_model(hidden_nodes_cnt=100, print_loss=True)
