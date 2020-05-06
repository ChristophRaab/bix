import unittest
import numpy as np
from scipy.optimize import minimize
from bix.utils.gsmo_solver import GSMO

class TESTGSMO(unittest.TestCase):
    def test_init_x(self):
        # Arrange
        x_solution = [0.75, 1.5, 0.75]

        # Act
        gsmo_solver = GSMO(A=np.zeros((3,3)),b=np.zeros((3,1)),C=C,d=d,r=0,R=5)

        # Assert
        for x1, x2 in tuple(zip(x_solution, gsmo_solver.x)):
            self.assertAlmostEqual(x1, x2)
