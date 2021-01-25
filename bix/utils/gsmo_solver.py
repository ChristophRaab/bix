import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import lsq_linear


class GSMO:
    def __init__(self, A, b, C=None, d=0, bounds=(None, None), optimization_type='minimize', max_iter=1000,
                 epsilon=1e-8,  # 1e-14
                 step_size=1):  # in the original way (SGD) it does not make any sense to have step size 1
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
            if np.count_nonzero(C) > 0:
                result = lsq_linear(C, d, bounds=(self.r, self.R))
                self.x = result.x
            else:
                self.x = np.random.default_rng().uniform(self.r, self.R, self.n)

            # if x is uniformly random between r-R -> C * x = D is not guaranteed
            test_res = C.dot(self.x)
            if not np.allclose(d, test_res):
                raise ValueError(
                    "The Equation Cx=d was not solvable. expected " + np.array_str(d) + " , got " + (np.array_str(
                        test_res) if isinstance(test_res, (np.ndarray)) else str(test_res)))

        else:
            self.C = np.zeros((1, self.n), dtype=float)
            # self.x = np.array([self.r] * self.n, dtype=float)
            self.x = np.random.default_rng().uniform(self.r, self.R, self.n)

        self.d = d

        # minimize or maximize
        self.optimization_type = optimization_type

        # size of working set
        self.K = np.linalg.matrix_rank(self.C) + 1
        # initial gradient
        self.gradient = (self.A + self.A.transpose()).dot(
            self.x) + self.b  # (b is left out later on in the update since x is updated only)

        # options
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.step_size = step_size

    def solve(self):
        for t in range(self.max_iter):
            S = self.__init_working_set()
            dF_best = 0
            j_best = -1
            j_all = np.linspace(0, self.n - 1, self.n).astype(int)
            j_without_S = np.setdiff1d(j_all, S)
            dx_best_S_best = 0
            for j in j_without_S:
                S.append(j)
                S = sorted(S)
                dx_S_best = self.__solve_small_QP(S)
                # dF_Temp is correct (checked with matlab FMS)
                dF_temp = abs(
                    dx_S_best.transpose().dot(self.A[:, S].transpose()[:, S].transpose()).dot(
                        dx_S_best) + self.gradient[S].transpose().dot(
                        dx_S_best))
                if dF_temp > dF_best:
                    dF_best = dF_temp
                    j_best = j
                    dx_best_S_best = dx_S_best
                S.remove(j)

            if dF_best < self.epsilon:
                print("Delta F < EPSILON")
                print(f'after {t + 1} iterations')
                print(f'with last delta gradient: {dF_best}')
                return self.x

            S.append(j_best)
            S = sorted(S)
            self.x[S] += self.step_size * dx_best_S_best

            grad_test = (self.A + self.A.transpose()).dot(self.x) + self.b
            # self.gradient += self.step_size * 2 * ((self.A + self.A.T + np.diag(self.b))[:, S] @ (dx_best_S_best))
            self.gradient += self.step_size * (self.A[:, S].dot(dx_best_S_best) + self.A[:, S].dot(dx_best_S_best))

        print("Max Iter reached")
        return self.x

    # first K - 1 Elements
    def __init_working_set(self):
        working_set = []
        active_set = []
        inactive_set = []
        gradient_displaced = np.empty((self.n,), dtype=[('idx', int), ('val', float)])
        for i in range(self.n):
            grad_at_i = self.gradient[i]
            w_best = self.__find_optimal_gradient_displacement(self.x[i], grad_at_i)
            gradient_displaced[i] = (i, abs((w_best - self.x[i]) * grad_at_i))

            if not gradient_displaced[i][1] == 0:
                active_set.append(i)
            else:
                inactive_set.append(i)
        if len(active_set) > self.K - 1:
            # p random indices only when K > 10
            p_upper_bound = round(len(active_set) * 0.1)
            p = 0
            if np.random.rand(1)[0] > 0.9:
                p = p_upper_bound

            sorted_v = np.sort(gradient_displaced, order='val')
            # reverse ordering
            sorted_v = sorted_v[::-1]
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
        choice_r = (self.r - x_i) * df_i  # at Eq 5
        choice_R = (self.R - x_i) * df_i  # at Eq 5
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
        if self.K == 1:
            dx_s = np.zeros((len(S), 1))
            k = S[0]
            alpha_min = self.r - self.x[k]
            alpha_max = self.R - self.x[k]
            beta = self.A[k, k]
            gamma = (self.gradient + self.b)[k]
            dx_s[0] = (self.__solve_bounded_second_degree(alpha_min, alpha_max, beta, gamma))
            return dx_s.reshape((dx_s.shape[0],))

        if self.K == 2:
            l = S[1]
            k = S[0]
            c_k = self.C[:, k][0]
            c_l = self.C[:, l][0]
            alpha_l = 0
            alpha_k = 0
            if (not c_k == 0) and (not c_l == 0):
                w, W = (self.r, self.R) if c_k * c_l > 0 else (self.R, self.r)
                alpha_min = max(self.r - self.x[l], ((self.x[k] - W) * c_k) / c_l)
                alpha_max = min(self.R - self.x[l], ((self.x[k] - w) * c_k) / c_l)
                beta = (((c_l * c_l) / (c_k * c_k)) * self.A[k, k]) + self.A[l, l] - (
                        (c_l / c_k) * (self.A[k, l] + self.A[l, k]))
                gamma = ((-c_l / c_k) * self.gradient[k]) + self.gradient[l]
                alpha_l = self.__solve_bounded_second_degree(alpha_min, alpha_max, beta, gamma)
                alpha_k = - (alpha_l * c_l) / c_k

            elif c_k == 0 and not c_l == 0:
                # alpha_l = 0 and alpha_k from special_case
                alpha_min = self.r - self.x[k]
                alpha_max = self.R - self.x[k]
                beta = self.A[k, k]
                gamma = self.gradient[k]
                alpha_k = self.__solve_bounded_second_degree(alpha_min, alpha_max, beta, gamma)

            elif c_l == 0 and not c_k == 0:
                # alpha_k = 0 and alpha_l from special_case
                alpha_min = self.r - self.x[l]
                alpha_max = self.R - self.x[l]
                beta = self.A[l, l]
                gamma = self.gradient[l]
                alpha_l = self.__solve_bounded_second_degree(alpha_min, alpha_max, beta, gamma)

            else:  # c_l == 0 and c_k == 0
                betas = np.array([[self.A[k, k], self.A[k, l]], [self.A[l, k], self.A[l, l]]])
                gamma_1 = self.gradient[k]
                gamma_2 = self.gradient[l]
                alpha_min_1 = self.r - self.x[k]
                alpha_max_1 = self.R - self.x[k]
                alpha_min_2 = self.r - self.x[l]
                alpha_max_2 = self.R - self.x[l]
                a_1, a_2 = self.__solve_bounded_conic(betas, gamma_1, gamma_2, alpha_min_1, alpha_max_1,
                                                              alpha_min_2, alpha_max_2)
                alpha_k = a_1
                alpha_l = a_2

            return np.array([alpha_k, alpha_l], dtype=float).reshape((2,))

        # general case K > 2
        u_k = null_space(self.C[:, S])
        dx_s = self.__solve_general_case(S, u_k)
        if dx_s is None:
            raise RuntimeError("Small QP was not solvable")

        return dx_s.reshape((dx_s.shape[0],))

    def __solve_bounded_second_degree(self, alpha_min, alpha_max, beta, gamma):
        # alpha = -(gamma)/2*beta if -(gamma)/2*beta in bounds amin,amax and beta < 0 (maximization) > 0 (minimization)
        if beta == 0:
            print(f'beta is 0 - return {alpha_min}')
            return alpha_min

        alpha = - gamma / (2 * beta)  # analytical solution (ok)
        if alpha_min <= alpha <= alpha_max and ((self.optimization_type == 'maximize' and beta < 0) or (
                self.optimization_type == 'minimize' and beta > 0)):
            return alpha
        else:
            # calculate the cost by setting dummy (valid) values for alpha (makes sense) (ok)
            cost_amin = (beta * alpha_min * alpha_min) + gamma * alpha_min
            cost_amax = (beta * alpha_max * alpha_max) + gamma * alpha_max
            if self.optimization_type == 'maximize' and cost_amax >= cost_amin:
                return alpha_max
            elif self.optimization_type == 'maximize' and cost_amin >= cost_amax:
                return alpha_min
            elif self.optimization_type == 'minimize' and cost_amin <= cost_amax:
                return alpha_min
            else:
                return alpha_max

    def __solve_bounded_conic(self, betas, gamma_1, gamma_2, alpha_min_1, alpha_max_1, alpha_min_2,
                              alpha_max_2):
        # solutions at the boundaries alpha_min_1, alpha_max_1, alpha_min_2, alpha_max_2
        solutions = []
        alpha_1 = alpha_min_1
        alpha_2 = self.__solve_bounded_second_degree(alpha_min_2, alpha_max_2, betas[1, 1],
                                                     gamma_2 + alpha_1 * (betas[0, 1] + betas[1, 0]))
        solutions.append((alpha_1, alpha_2))

        alpha_1 = alpha_max_1
        alpha_2 = self.__solve_bounded_second_degree(alpha_min_2, alpha_max_2, betas[1, 1],
                                                     gamma_2 + alpha_1 * (betas[0, 1] + betas[1, 0]))
        solutions.append((alpha_1, alpha_2))

        alpha_2 = alpha_min_2
        alpha_1 = self.__solve_bounded_second_degree(alpha_min_1, alpha_max_1, betas[0, 0],
                                                     gamma_1 + alpha_2 * (betas[0, 1] + betas[1, 0]))
        solutions.append((alpha_1, alpha_2))

        alpha_2 = alpha_max_2
        alpha_1 = self.__solve_bounded_second_degree(alpha_min_1, alpha_max_1, betas[0, 0],
                                                     gamma_1 + alpha_2 * (betas[0, 1] + betas[1, 0]))
        solutions.append((alpha_1, alpha_2))

        # if (B_12 + B_21)^2 - 4 * B_11 * B_22 < 0 then check if optimum inside 2 * Betas * alphas = - Gammas
        if (betas[0, 1] - betas[1, 0]) * (betas[0, 1] - betas[1, 0]) - 4 * betas[0, 0] * betas[1, 1] < 0:
            gammas = np.array([gamma_1, gamma_2])
            result = lsq_linear(2 * betas, gammas)
            solutions.append((result.x[0], result.x[1]))

        optimum = float('inf') if self.optimization_type == 'minimize' else float('-inf')
        best_alphas = ()
        for a_1, a_2 in solutions:
            value = (betas[0, 0] * a_1 * a_1) \
                    + (betas[1, 1] * a_2 * a_2) \
                    + ((betas[0, 1] + betas[1, 0]) * a_1 * a_2) \
                    + (gamma_1 * a_1) \
                    + (gamma_2 * a_2)

            if self.optimization_type == 'minimize':
                if value < optimum:
                    optimum = value
                    best_alphas = (a_1, a_2)
            else:
                if value > optimum:
                    optimum = value
                    best_alphas = (a_1, a_2)

        return best_alphas

    def __solve_general_case(self, S, Us):
        # q = ((self.A.dot(self.x) + self.A.T.dot(self.x) +  self.b)[S]).transpose().dot(Us)
        q = (self.gradient[S]).transpose().dot(Us)  # no + self.b here - does not work
        D = len(q)
        P = Us.transpose().dot(self.A[np.ix_(S, S)]).dot(Us)
        G = np.vstack((-Us, Us))
        D2 = len(S)
        h = np.zeros((2 * D2))
        for i in range(D2):
            lb = -self.r  # to get the <= as > we have to change both signs
            ub = self.R
            h[i] = lb + self.x[S[i]]
            h[D2 + i] = ub - self.x[S[i]]
        x = cp.Variable(D)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x), [G @ x <= h])

        prob.solve()
        xRes = Us.dot(x.value)
        return xRes  # x.value #x.value
