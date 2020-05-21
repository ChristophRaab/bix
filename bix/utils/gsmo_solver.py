import numpy as np
from scipy.linalg import null_space
from scipy.optimize import lsq_linear
import cvxpy as cp


class GSMO:
    def __init__(self, A, b, C=None, d=0, bounds=(None, None), optimization_type='minimize', max_iter=10000,
                 epsilon=0.0001,
                 step_size=1):
        # optimize F: x'Ax + b'x  s.t.  Cx=d, x elements [r,R]^n
        self.A = A
        self.b = b
        # number of components
        self.n = A.shape[1]
        # lower bound
        if bounds[0] is not None:
            self.r = bounds[0]
        else:
            self.r = -10000
        # upper bound
        if bounds[1] is not None:
            self.R = bounds[1]
        else:
            self.R = 10000
        if C is not None:
            self.C = C
            # first guess such that Cx = d and x elements [r,R]^n
            result = lsq_linear(C, d, bounds=(self.r, self.R))
            self.x = result.x
            test_res = C.dot(self.x)
            if not np.allclose(d, test_res):
                raise ValueError(
                    "The Equation Cx=d was not solvable. expected " + np.array_str(d) + " , got " + np.array_str(
                        test_res))
            # bounds?
        else:
            self.C = np.zeros((1, self.n), dtype=float)
            self.x = np.array([self.r] * self.n, dtype=float)
        self.d = d

        # minimize or maximize
        self.optimization_type = optimization_type

        # size of working set
        self.K = np.linalg.matrix_rank(self.C) + 1
        # initial gradient
        self.gradient = (self.A + self.A.transpose()).dot(self.x) + self.b

        # options
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.step_size = step_size

    def solve(self):
        for t in range(self.max_iter):
            S = self.__init_working_set()
            dF_best = 0
            j_best = -1
            dx_best_S_best = any
            j_all = [i for i in range(self.n)]
            j_without_S = np.setdiff1d(j_all, S)
            for j in j_without_S:
                S.append(j)
                S = sorted(S)
                dx_S_best = self.__solve_small_QP(S)
                dF_temp = abs(
                    dx_S_best.transpose().dot(self.A[:, S].transpose()[:, S].transpose()).dot(
                        dx_S_best) + self.gradient[S].transpose().dot(
                        dx_S_best))
                if dF_temp > dF_best:
                    dF_best = dF_temp
                    j_best = j
                    dx_best_S_best = dx_S_best
                S.remove(j)

            if j_best == -1:
                raise RuntimeError("cant find second best choice to optimise")

            S.append(j_best)
            self.x[S] += self.step_size * dx_best_S_best
            self.gradient += self.step_size * (self.A + self.A.transpose() + np.diag(self.b))[:, S].dot(dx_best_S_best)

            if dF_best < self.epsilon:
                print("Delta F < EPSILON")
                print(f'after iterations: {t + 1}')
                print(f'with last delta gradient: {dF_best}')
                return self.x

        print("Max Iter reached")
        return self.x

    # first K - 1 Elements
    def __init_working_set(self):
        working_set = []
        active_set = []
        inactive_set = []
        gradient_displaced = np.empty((self.n,), dtype=[('idx', int), ('val', float)])
        for i in range(self.n):
            w_best = self.__find_optimal_gradient_displacement(self.x[i], self.gradient[i])
            gradient_displaced[i] = (i, abs((w_best - self.x[i]) * self.gradient[i]))

            if not gradient_displaced[i][1] == 0:
                active_set.append(i)
            else:
                inactive_set.append(i)
        if len(active_set) > self.K - 1:
            # p random indices only when K > 10
            p_upper_bound = round(len(active_set) * 0.1)
            p = np.random.choice([i for i in range(p_upper_bound + 1)], 1)[0]

            sorted_v = np.sort(gradient_displaced, order='val')
            for i in range(self.K - p - 1):
                working_set.append(sorted_v[i][0])

            active_without_working_set = np.setdiff1d(active_set, working_set)
            working_set.extend(np.random.choice(active_without_working_set, p))

        else:
            working_set.extend(active_set)
            random_idx_count = self.K - 1 - len(active_set)
            working_set.extend(np.random.choice(inactive_set, random_idx_count))
        if len(working_set) > 0:
            working_set = sorted(working_set)

        return working_set

    def __find_optimal_gradient_displacement(self, x_i, df_i):
        choice_r = (self.r - x_i) * df_i
        choice_R = (self.R - x_i) * df_i
        if self.optimization_type == 'maximize':
            # QP is maximized we pick n_i* such that n_i* x df_i >= 0
            # n_i = (w - x_i)
            if choice_r >= choice_R:
                return self.r
            else:
                return self.R
        else:
            # OP is minimized we pick n_i* such that n_i* x df_i <= 0
            # n_i = (w - x_i)
            if choice_r <= choice_R:
                return self.r
            else:
                return self.R

    def __solve_small_QP(self, S):
        """if self.K == 1:
            k = S[0]
            alpha_min = self.r - self.x[k]
            alpha_max = self.R - self.x[k]
            beta = self.A[k, k]
            gamma = self.gradient[k]
            return self.solve_special_case(alpha_min, alpha_max, beta, gamma, k)
        elif self.K == 2:
            l = S[1]
            k = S[0]
            c_k = self.C[:, k][0]
            c_l = self.C[:, l][0]
            if (not c_k == 0) and (not c_l == 0):
                w, W = (self.r, self.R) if c_k * c_l > 0 else (self.R, self.r)
                alpha_min = max(self.r - self.x[l], ((self.x[k] - W) * c_k) / c_l)
                alpha_max = min(self.R - self.x[l], ((self.x[k] - w) * c_k) / c_l)
                beta = self.A[l, l]
                gamma = self.gradient[l]
                alpha_l = self.solve_special_case(alpha_min, alpha_max, beta, gamma, l)
                alpha_k = - (alpha_l * c_l) / c_k
                return np.array([alpha_k, alpha_l]).reshape((2,))
        else:"""
        u_k = null_space(self.C[:, S])
        alpha_k = self.__find_optimal_solution(S, np.linalg.matrix_rank(self.C) - np.linalg.matrix_rank(
            self.C[:, S]) + 1)
        if alpha_k is None:
            raise RuntimeError("Small QP was not solvable")

        dx_s = np.zeros((u_k.shape[0], 1))
        for idx, a in np.ndenumerate(alpha_k):
            dx_s += a * u_k[:, idx]
        return dx_s.reshape((dx_s.shape[0],))

    def solve_special_case(self, alpha_min, alpha_max, beta, gamma, k):
        # alpha = -(gamma)/2*beta if -(gamma)/2*beta in bounds amin,amax and beta < 0 (maximization) > 0 (minimization)

        alpha = - gamma / 2 * beta
        if alpha_min <= alpha <= alpha_max and ((self.optimization_type == 'maximize' and beta < 0) or (
                self.optimization_type == 'minimize' and beta > 0)):
            return alpha
        else:
            cost_amin = beta * alpha_min * alpha_min + gamma * alpha_min
            cost_amax = beta * alpha_max * alpha_max + gamma * alpha_max
            if self.optimization_type == 'maximize' and cost_amax >= cost_amin:
                return alpha_max
            elif self.optimization_type == 'maximize' and cost_amin >= cost_amax:
                return alpha_min
            elif self.optimization_type == 'minimize' and cost_amin <= cost_amax:
                return alpha_min
            else:
                return alpha_max

    def __find_optimal_solution(self, S, D):
        bounds = []
        G = np.zeros((2 * D, D))
        h = np.zeros((2 * D))
        cond_idx = 0
        for i in range(D):
            lb, ub = self.__get_bounds(self.x[i])
            G[cond_idx, i] = 1
            h[cond_idx] = ub
            G[cond_idx + 1, i] = -1
            h[cond_idx + 1] = -lb
            cond_idx += 2
            bounds.append((lb, ub))

        S_d = S[:D]

        P = self.A[:, S_d].transpose()[:, S_d].transpose()
        q = self.gradient[S_d]
        x = cp.Variable(D)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                         [G @ x <= h])
        prob.solve()

        return x.value

    def __get_bounds(self, x_i):
        a_min = self.r - x_i
        a_max = self.R - x_i
        return a_min, a_max


def objective_function(a, D, S, A, grad):
    sum1 = 0
    for k in range(D):
        sum1 += (a[k] * a[k]) * A[S[k], S[k]]

    sum2 = 0
    for k in range(D):
        for i in range(D):
            if not i == k:
                sum2 += (a[k] * a[i]) * A[S[i], S[k]]

    sum3 = 0
    for k in range(D):
        sum3 += a[k] * grad[S[k]]

    return sum1 + sum2 + sum3
