import unittest
import numpy as np
from . import knn_classifier


class KnnClassifierTests(unittest.TestCase):

    def test_classification_by_knn(self):
        x_in = np.array([[1, 2, 3],
                         [-3, -2, -1],
                         [-1, 0, -1]])
        y_in = np.array(['A', 'B', 'B'])
        x_test = np.array([[4, 5, 6],
                           [-4, -5, -6]])
        y_test = np.array(['A', 'B'])

        y_pred = knn_classifier.classify(x_in, y_in, x_test, k=1)

        np.testing.assert_equal(y_test, y_pred)
