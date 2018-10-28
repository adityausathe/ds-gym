import numpy as np


def predict(x_in, weights):
    logit = combine_logistically(x_in, weights)
    return classify_logits(logit)


def classify_logits(logit):
    return (logit > 0.5).astype(int)


def combine_logistically(x_in, weights):
    lin_out = x_in.dot(weights)
    logit = 1 / (1 + np.exp(-lin_out))
    return logit


def train(x_in, y_in, l_rate=0.1, max_epochs=5):
    """
    :param x_in: m x p matrix containing m data points
    :param y_in: m x 1 vector containing classes of m data points
    :return:
    """
    m, n_features = x_in.shape
    weights = np.zeros(n_features)

    epochs_elapsed = 0
    while epochs_elapsed < max_epochs:
        y_logit = combine_logistically(x_in, weights)
        loss = - (y_in.T.dot(np.log(y_logit)) + (1 - y_in).T.dot(np.log(1 - y_logit))) / m
        print(loss)

        nabla_w = x_in.T.dot(y_logit - y_in) / m
        weights -= l_rate * nabla_w
        print(nabla_w)

        epochs_elapsed += 1
    print(weights)
    return weights
