import numpy as np

from pySDC.implementations.sweeper_classes.Runge_Kutta import (
    ButcherTableau,
    RungeKutta,
    BackwardEuler,
    ImplicitMidpointMethod,
)


class RungeKuttaDAE(RungeKutta):
    r"""
    This class is the base class for Runge-Kutta methods to solve differential-algebraic equations (DAEs).

    Note
    ----
    As different from the ODE case ``level.f`` stores the gradient of the solution, *not* the values of
    the right-hand side of the problem.
    """

    def __init__(self, params):
        super().__init__(params)
        self.f_init = None
        print(self.coll.Qmat)
        print(self.coll.nodes)

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep
        """

        # get current level and problem
        L = self.level
        P = L.prob

        f_init = self.f_init if self.f_init is not None else P.du_exact(L.time)
        L.f[0] = P.dtype_f(f_init)  # P.dtype_f(init=P.init, val=0.0)
        for m in range(1, self.coll.num_nodes + 1):
            L.u[m] = P.dtype_u(init=P.init, val=0.0) # P.dtype_u(L.u[0])
            L.f[m] = P.dtype_f(init=P.init, val=0.0) # P.dtype_f(f_init)

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True
        print(L.time, 'Predict:', 'u:', L.u)
        print(L.time, 'Predict:', 'f:', L.f)
        print('u0 in predict:', L.u[0])
        print()

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * L.f[j]

        return me

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        assert L.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes
        u0 = L.u[0]
        print('M:', M)
        print('all nodes in a step:', [L.time + L.dt * self.coll.nodes[m] for m in range(M)])
        for m in range(0, M):
            u_approx = P.dtype_u(u0)
            for j in range(1, m + 1):
                # print('m:', m, 'j:', j)
                u_approx += L.dt * self.QI[m + 1, j] * L.f[j]
            print('m:', m, [self.QI[m + 1, j] for j in range(1, m + 1)])
            def implSystem(unknowns):
                """
                Build implicit system to solve in order to find the unknowns for the derivative
                of u.

                Parameters
                ----------
                unknowns : dtype_u
                    Unknowns of the system.

                Returns
                -------
                sys : dtype_f
                    System to be solved.
                """
                unknowns_mesh = P.dtype_f(P.init)
                unknowns_mesh[:] = unknowns

                local_u_approx = u_approx

                # defines the "implicit" factor, note that for explicit RK the diagonal element is zero
                local_u_approx += L.dt * self.QI[m + 1, m + 1] * unknowns_mesh
                print('diagonal element m:', m, self.QI[m + 1, m + 1], local_u_approx)
                sys = P.eval_f(local_u_approx, unknowns_mesh, L.time + L.dt * self.coll.nodes[m])
                print('sys:', sys)
                return sys

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            du_new = P.solve_system(implSystem, L.f[m], L.time + L.dt * self.coll.nodes[m])

            L.f[m + 1][:] = du_new

        # Update numerical solution
        integral = self.integrate()
        for k in range(M):
            L.u[k + 1] = u0 + integral[k]

        self.f_init = L.f[-1]
        print('After update:', 'u:', L.u)
        print('After update:', 'f:', L.f)
        print()

        # indicate presence of new values at this level
        L.status.updated = True

        return None


class RungeKuttaIMEXDAE(RungeKuttaDAE):
    r"""
    This is an IMEX base class for implementing Runge-Kutta methods for DAEs, where only the differential
    variables will be integrated. This is useful for DAEs of semi-explicit form.
    """

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1][: P.diff_nvars] += L.dt * self.coll.Qmat[m, j] * L.f[j][: P.diff_nvars]
                me[-1][P.diff_nvars :] += L.u[j][P.diff_nvars :]
                print()

        return me

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        assert L.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes
        u0 = L.u[0]
        print('M:', M)
        for m in range(0, M):
            u_approx = P.dtype_u(u0)  # indices correct here?
            for j in range(1, m + 1):
                u_approx += L.dt * self.QI[m + 1, j] * L.f[j]
            u_approx[P.diff_nvars :] = L.u[m + 1][P.diff_nvars :]
            print('u0 before implSystem:', L.u[0])
            print('m:', m, [self.QI[m + 1, j] for j in range(1, m + 1)])
            def implSystem(unknowns):
                """
                Build implicit system to solve in order to find the unknowns for the derivative
                of u.

                Parameters
                ----------
                unknowns : dtype_u
                    Unknowns of the system.

                Returns
                -------
                sys : dtype_f
                    System to be solved.
                """
                unknowns_mesh = P.dtype_f(P.init)
                unknowns_mesh[:] = unknowns

                local_u_approx = u_approx

                # defines the "implicit" factor, note that for explicit RK the diagonal element is zero
                local_u_approx += L.dt * self.QI[m + 1, m + 1] * unknowns_mesh
                local_u_approx[P.diff_nvars :] = unknowns_mesh[P.diff_nvars :]
                print('diagonal element m:', m, self.QI[m + 1, m + 1], local_u_approx)
                sys = P.eval_f(local_u_approx, unknowns_mesh[: P.diff_nvars], L.time + L.dt * self.coll.nodes[m])
                print('sys:', sys)
                return sys

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            U0_diff, p0_alg = np.array(L.f[m][: P.diff_nvars]), np.array(L.u[m][P.diff_nvars :])
            du0 = np.concatenate((U0_diff, p0_alg))
            du_new = P.solve_system(implSystem, du0, L.time + L.dt * self.coll.nodes[m])

            L.f[m + 1][: P.diff_nvars] = du_new[: P.diff_nvars]
            L.u[m + 1][P.diff_nvars :] = du_new[P.diff_nvars :]

        # Update numerical solution
        integral = self.integrate()
        for k in range(M):
            L.u[k + 1][: P.diff_nvars] = L.u[0][: P.diff_nvars] + integral[k][: P.diff_nvars]

        # store value at last node as initial condition for next step
        self.f_init = L.f[-1]
        print('After update:', 'u:', L.u)
        print('After update:', 'f:', L.f)

        # indicate presence of new values at this level
        L.status.updated = True

        return None


class DIRK43_2(RungeKutta):
    """
    L-stable Diagonally Implicit RK method with four stages of order 3.
    Taken from https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    nodes = np.array([0.5, 2.0 / 3.0, 0.5, 1.0])
    weights = np.array([3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5])
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 0.5
    matrix[1, :2] = [1.0 / 6.0, 0.5]
    matrix[2, :3] = [-0.5, 0.5, 0.5]
    matrix[3, :] = [3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5]
    ButcherTableauClass = ButcherTableau


class BackwardEulerDAE(RungeKuttaDAE, BackwardEuler):
    pass

class ImplicitMidpointMethodDAE(RungeKuttaDAE, ImplicitMidpointMethod):
    pass

class ImplicitMidpointMethodIMEXDAE(RungeKuttaIMEXDAE, ImplicitMidpointMethod):
    pass

class DIRK43_2DAE(RungeKuttaDAE, DIRK43_2):
    pass

class DIRK43_2IMEXDAE(RungeKuttaIMEXDAE, DIRK43_2):
    pass
