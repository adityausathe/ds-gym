import unittest
import numpy as np

from . import logistic_classifier;


class LogisticClassifierTests(unittest.TestCase):

    def test_predict_logistically(self):
        x_in = np.array([[1, 2],
                         [-2, -1]])
        weights = np.array([1, 1])
        y_exp = np.array(([1, 0]))

        y_out = logistic_classifier.predict(x_in, weights)

        np.testing.assert_array_equal(y_exp, y_out)

    def test_SGD_training_of_weights(self):
        x_in = np.array([[1, 2, 9],
                         [-2, -1, 9]])
        y_in = np.array([1, 0])

        learned_weights = logistic_classifier.train(x_in, y_in, l_rate=0.01, max_epochs=10)

        x_test = np.array([[5, 0, 0],
                           [-4, -1, 0]])
        y_test = np.array([1, 0])
        y_pred = logistic_classifier.predict(x_test, learned_weights)

        np.testing.assert_equal(y_test, y_pred)
