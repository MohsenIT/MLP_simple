import numpy as np

import helper.data_preparation as dt

'''DATA PREPARATION'''
dim = 3
train_data, train_target = dt.generate_data(cnt=500, dim=dim, low=1, high=6)
test_data, test_target = dt.generate_data(cnt=200, dim=dim, low=1.5, high=5.5)

''''INITIALIZATION: weight init and shuffling data'''
train_size = len(train_target)  # training set size
nn_input_dim = 3  # input layer dimensionality
nn_output_dim = 1  # output layer dimensionality

# Gradient descent parameters
learning_rate = 0.000001  # learning rate for gradient descent
reg_lambda = 0.01  # regularization strength

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
    return z1, a1, z2, a2, scale_res(a2)


def scale_res(r):
    return r * (30 - 4.9) + 4.9


# evaluate the total loss between target and output considering weights norm
def calculate_loss(output, target, weights):
    w1, w2 = weights['w1'], weights['w2']
    target_loss_term = np.mean(np.square(output - target))
    weight_regularization_term = reg_lambda * 0.5 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return target_loss_term + weight_regularization_term


# this function learns parameters for the neural network and returns the model.
def build_model(hidden_nodes_cnt, num_passes=2000, print_loss=False):
    # initialize the parameters to random values.
    np.random.seed(0)
    w1 = np.random.randn(nn_input_dim, hidden_nodes_cnt) / np.sqrt(nn_input_dim)
    b1 = np.random.randn(hidden_nodes_cnt) / np.sqrt(hidden_nodes_cnt)
    w2 = np.random.randn(hidden_nodes_cnt, nn_output_dim) / np.sqrt(hidden_nodes_cnt)
    b2 = np.random.randn(nn_output_dim) / np.sqrt(nn_output_dim)
    current_model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # gradient descent. For each batch...
    for i in range(0, num_passes):
        # call
        z1, a1, z2, a2, output = predict(current_model, x)
        if print_loss:
            if i == 0: print("iteration\ttrain MSE\ttest MSE")
            train_mse = np.sqrt(np.mean(np.square(output - y)))
            _, _, _, _, test_output = predict(current_model, test_data)
            test_mse = np.sqrt(np.mean(np.square(test_output - test_target)))
            print("%i\t%f\t%f" % (i + 1, train_mse, test_mse))

        # recall
        derv_out2 = (y - output) * (output * (1 - output)) * (30 - 4.9)
        delta2 = np.matmul(np.transpose(a1), derv_out2) / train_size
        dw2 = delta2 + reg_lambda * w2
        db2 = np.mean(b2 * derv_out2, 0)

        derv_out1 = a1 * np.reshape(np.sum(delta2 * w2, 1), [1, a1.shape[1]])
        delta1 = np.matmul(np.transpose(x), derv_out1) / train_size
        dw1 = delta1 + reg_lambda * w1
        db1 = np.mean(b1 * derv_out1, 0)

        # gradient descent parameter update
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

        # Assign new parameters to the model
        current_model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return current_model


'''a) LEARNING'''
# Build a model with a 3-dimensional hidden layer
model = build_model(hidden_nodes_cnt=10, num_passes=500, print_loss=True)
