import numpy as np
from numpy.linalg import inv


def combine_linearly(x_in, weights):
    return np.matmul(np.transpose(weights), x_in)


def do_closed_regression(x_train, y_train):
    # Applying direct closed loop solution for linear model formulated by given training data
    # Of course, considering MSE cost
    learned_weights = inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    return learned_weights