import unittest
import numpy as np

from . import linear_regressor


class LinearRegressorTests(unittest.TestCase):

    def test_linear_combiner(self):
        x_in = np.array([[1, 2],
                         [2, 1]])
        weights = np.array([[1], [1]])
        exp_y_out = np.array(([[3, 3]]))
        y_out = linear_regressor.combine_linearly(x_in, weights)

        np.testing.assert_array_equal(exp_y_out, y_out)

    def test_linear_regression_using_closed_opt(self):
        x_train = np.array([[1.5, 2],
                            [2, 9]])
        y_train = np.array(([[3.5], [11]]))

        learned_weights = linear_regressor.do_closed_regression(x_train, y_train)

        x_test = np.array([[1, 2],
                           [2, 1]])
        exp_y_test = np.array(([[3, 3]]), dtype=float)
        y_test = linear_regressor.combine_linearly(x_test, learned_weights)
        np.testing.assert_array_almost_equal(exp_y_test, y_test)
