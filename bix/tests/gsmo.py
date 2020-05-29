import os
import unittest

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from bix.utils.gsmo_solver import GSMO


class TESTGSMO(unittest.TestCase):
    def test_init_x_1d(self):
        # Arrange
        C = np.array([-1, 1, -1])
        d = 3

        # Act
        gsmo_solver = GSMO(A=np.zeros((3, 3)), b=np.zeros((3, 1)), C=C, d=d, bounds=(0, 5))

        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        self.assertAlmostEqual(d, result)

    def test_init_x_2d(self):
        # Arrange
        C = np.array([[-1, 1, -1], [2, 0, 3]])
        d = np.array([3, 1])

        # Act
        gsmo_solver = GSMO(A=np.zeros((3, 3)), b=np.zeros((3, 1)), C=C, d=d, bounds=(0, 5))

        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        np.testing.assert_almost_equal(d, result)

    def test_init_x_valueError(self):
        # Arrange
        C = np.array([[-1, 1, 1], [-1, 1, 1]])
        d = np.array([2, 3])

        # Act
        with self.assertRaises(ValueError):
            GSMO(A=np.zeros((3, 3)), b=np.zeros((3, 1)), C=C, d=d, bounds=(0, 5))

    def test_init_small_svm(self):
        # Arrange
        pwd = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(pwd, "small_svm_problem_data.csv")
        data = pd.read_csv(test_data_file, delimiter=',')
        print(data)
        A = np.zeros((data.shape[0], data.shape[0]))
        points = data[['X', 'Y']]
        y = data['Label']
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                A[i, j] = y.iloc[i] * y.iloc[j] * points.iloc[i].dot(points.iloc[j])

        A = (-0.5) * A
        b = np.ones((1, A.shape[0]))
        C = y.to_numpy()
        d = 0

        # Act
        gsmo_solver = GSMO(A, b, C, d, bounds=(0, 100))

        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        np.testing.assert_almost_equal(d, result)

    def test_solve_small_svm(self):
        # Arrange
        pwd = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(pwd, "small_svm_problem_data.csv")
        data = pd.read_csv(test_data_file, delimiter=',')
        print(data)
        A = np.zeros((data.shape[0], data.shape[0]), dtype=float)
        points = data[['X', 'Y']]
        y = data['Label']
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                A[i, j] = y.iloc[i] * y.iloc[j] * points.iloc[i].dot(points.iloc[j])

        A = (0.5) * A
        b = -np.ones((A.shape[0],), dtype=float)
        C = y.to_numpy().astype(dtype=float)
        C_t = C.reshape((1, C.shape[0]))
        d = np.zeros((1,), dtype=float)
        lb = 0
        ub = 1
        gsmo_solver = GSMO(A, b, C_t, d, bounds=(lb, ub), step_size=0.001)

        clf = SVC(C=1, kernel='linear')
        clf.fit(points, y)

        # Act
        print("#### SMO  ####")
        gsmo_solver.solve()
        print(gsmo_solver.x)

        G = np.zeros((2 * A.shape[0], A.shape[0]))
        h = np.zeros((2 * A.shape[0]))
        cond_idx = 0
        for i in range(A.shape[0]):
            G[cond_idx, i] = 1
            h[cond_idx] = ub
            G[cond_idx + 1, i] = -1
            h[cond_idx + 1] = -lb
            cond_idx += 2

        x = cp.Variable(A.shape[0])
        b = -b
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, A) - b.T @ x),
                          [G @ x <= h, C @ x == d])
        prob.solve()
        print("\n#### CVXPY ####")
        print(x.value)

        print("\n#### SVC ####")
        print(clf.dual_coef_)
        print(clf.support_)

        plt.scatter(points['X'], points['Y'], c=y)
        plt.scatter(points['X'].iloc[clf.support_], points['Y'].iloc[clf.support_], c='r')
        plt.show()

        # Assert
        # Cx = d
        result = C.dot(gsmo_solver.x)
        np.testing.assert_almost_equal(d, result)
        # np.testing.assert_almost_equal(gsmo_solver.x, res.x)

    def test_small_qp_without_constraints(self):
        # Arrange
        A = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.array([1, -1], dtype=float).reshape((2,))
        lb = -1
        ub = 1
        gsmo_solver = GSMO(A=A, b=b, bounds=(lb, ub), step_size=1)

        x = cp.Variable(A.shape[0])
        G = np.zeros((2 * A.shape[0], A.shape[0]))
        h = np.zeros((2 * A.shape[0]))
        cond_idx = 0
        for i in range(A.shape[0]):
            G[cond_idx, i] = 1
            h[cond_idx] = ub
            G[cond_idx + 1, i] = -1
            h[cond_idx + 1] = -lb
            cond_idx += 2
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, A) + b.T @ x),
                          [G @ x <= h])
        prob.solve()
        print("\n#### CVXPY ####")
        print(x.value)

        # Act
        print("#### SMO  ####")
        gsmo_solver.solve()
        print(gsmo_solver.x)

        # Assert
        np.testing.assert_almost_equal(gsmo_solver.x, x.value, 5)

    def test_small_qp_with_constraints(self):
        # Arrange
        A = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.array([1, -1], dtype=float).reshape((2,))
        C = np.array([[1, 1]], dtype=float)
        d = np.array([7])
        lb = -10
        ub = 10
        gsmo_solver = GSMO(A=A, b=b, C=C, d=d, bounds=(lb, ub), step_size=0.01)

        x = cp.Variable(A.shape[0])
        G = np.zeros((2 * A.shape[0], A.shape[0]))
        h = np.zeros((2 * A.shape[0]))
        cond_idx = 0
        for i in range(A.shape[0]):
            G[cond_idx, i] = 1
            h[cond_idx] = ub
            G[cond_idx + 1, i] = -1
            h[cond_idx + 1] = -lb
            cond_idx += 2
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, A) + b.T @ x),
                          [G @ x <= h, C @ x == d])
        prob.solve()
        print("\n#### CVXPY ####")
        print(x.value)

        # Act
        print("#### SMO  ####")
        gsmo_solver.solve()
        print(gsmo_solver.x)

        # Assert
        np.testing.assert_almost_equal(gsmo_solver.x, x.value, 3)


if __name__ == '__main__':
    unittest.main()
