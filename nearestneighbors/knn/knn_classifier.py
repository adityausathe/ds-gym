from collections import Counter

import numpy as np


def classify(x_in, y_in, x_test, k=1):
    test_size = x_test.shape[0]
    y_pred = np.empty(test_size, dtype=str)
    dist = np.zeros((x_in.shape[0], test_size))
    dist_classes = np.array([y_in] * test_size).T
    i = 0
    for x in x_in:
        dist[i] = np.sum(np.square(x_test - x), axis=1)
        i += 1

    sort_order = dist.argsort(axis=0)
    i = 0
    for x in x_test:
        k_nearest_neighbors = dist_classes[:, i][sort_order[:, i]][:k]
        y_pred[i], occ = Counter(k_nearest_neighbors).most_common(1)[0]
        print(y_pred[i], 'occurred', occ, 'times in the neighborhood of', x)
        i += 1
    return y_pred
