import unittest
import numpy as np
from bix.utils.libqp.libqp import LibQP


class LIBQP_TEST(unittest.TestCase):
    def test_simpleQP(self):
        test = LibQP(np.array([[2, 0], [0, 2]], dtype=float), np.array([1, -1], dtype=float),
                     np.array([1, 1], dtype=float), -5,
                     np.array([-10, -10], dtype=float), np.array([10, 10], dtype=float), np.array([1, 1], dtype=float),
                     100000,
                     0.0001)
        sl = test.solve()
        print(sl.nIter)
        print(sl.QP)
        print(sl.QD)
        print(sl.exitFlag)
        print(test.x)
        np.testing.assert_almost_equal(test.x, np.array([0.5, 1.5]), 5)
