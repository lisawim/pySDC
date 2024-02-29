import numpy as np

from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta, ButcherTableau, BackwardEuler


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
        self.du_init = None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep
        """

        # get current level and problem
        L = self.level
        P = L.prob

        du_init = self.du_init if self.du_init is not None else P.du_exact(L.time)
        print(du_init, self.coll.nodes)
        L.f[0] = P.dtype_f(du_init)
        for m in range(1, self.coll.num_nodes + 1):
            L.u[m] = P.dtype_u(init=P.init, val=0.0)
            L.f[m] = P.dtype_f(init=P.init, val=0.0)
        print(f'predict at time {L.time}:')
        print(L.f)
        print()
        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def integrate(self):
        """
        Integrates the right-hand side.

        Returns
        -------
        me : list of dtype_u
            Containing the integral as values.
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
        for m in range(0, M):
            u_approx = P.dtype_u(L.u[0])
            for j in range(1, m + 1):
                u_approx += L.dt * self.QI[m + 1, j] * L.f[j]

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
                unknowns_mesh = P.dtype_f(unknowns)

                local_u_approx = P.dtype_f(u_approx)

                # defines the "implicit" factor, note that for explicit RK the diagonal element is zero
                local_u_approx += L.dt * self.QI[m + 1, m + 1] * unknowns_mesh
                sys = P.eval_f(local_u_approx, unknowns_mesh, L.time + L.dt * self.coll.nodes[m + 1])
                return sys

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            du_new = P.solve_system(implSystem, L.f[m], L.time + L.dt * self.coll.nodes[m + 1])

            L.f[m + 1][:] = du_new

        # Update numerical solution
        integral = self.integrate()
        for m in range(M):
            L.u[m + 1] = L.u[0] + integral[m]
        # print('After solve:')
        # print(L.f)
        # print('Last node:')
        # print(L.f[-1])
        self.du_init = L.f[-1]

        # indicate presence of new values at this level
        L.status.updated = True

        return None


class EDIRK4(RungeKutta):
    """
    Stiffly accurate, fourth-order EDIRK with four stages. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), second one in eq. (216).
    """

    nodes = np.array([0.0, 3.0 / 2.0, 7.0 / 5.0, 1.0])
    weights = np.array([13.0, 84.0, -125.0, 70.0]) / 42.0
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 0
    matrix[1, :2] = [3.0 / 4.0, 3.0 / 4.0]
    matrix[2, :3] = [447.0 / 675.0, -357.0 / 675.0, 855.0 / 675.0]
    matrix[3, :] = [13.0 / 42.0, 84.0 / 42.0, -125.0 / 42.0, 70.0 / 42.0]
    # nodes = np.array([0, 1 / 2, 1])#np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    # weights = np.array([1/6, 4/6, 1/6])#np.array([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0])
    # matrix = np.zeros((3, 3))#np.zeros((4, 4))
    # matrix[0, 0] = 0
    # matrix[1, :2] = [1/4, 1/4]#[1.0 / 6.0, 1.0 / 6.0]
    # matrix[2, :3] = [1/6, 4/6, 1/6]#[1.0 / 12.0, 1.0 / 2.0, 1.0 / 12.0]
    # matrix[3, :] = [1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0]
    # nodes = np.array([0.5, 2.0 / 3.0, 0.5, 1.0])
    # weights = np.array([3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5])
    # matrix = np.zeros((4, 4))
    # matrix[0, 0] = 0.5
    # matrix[1, :2] = [1.0 / 6.0, 0.5]
    # matrix[2, :3] = [-0.5, 0.5, 0.5]
    # matrix[3, :] = [3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5]
    ButcherTableauClass = ButcherTableau


class TrapezoidalRule(RungeKutta):
    """
    Famous trapezoidal rule of second order. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), third one in eq. (213).
    """

    nodes = np.array([0.0, 1.0])
    weights = np.array([1.0 / 2.0, 1.0 / 2.0])
    matrix = np.zeros((2, 2))
    matrix[0, 0] = 0.0
    matrix[1, :] = [1.0 / 2.0, 1.0 / 2.0]
    ButcherTableauClass = ButcherTableau


class BackwardEulerDAE(RungeKuttaDAE, BackwardEuler):
    pass


class TrapezoidalRuleDAE(RungeKuttaDAE, TrapezoidalRule):
    pass


class EDIRK4DAE(RungeKuttaDAE, EDIRK4):
    pass