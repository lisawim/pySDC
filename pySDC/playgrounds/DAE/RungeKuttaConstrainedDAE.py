import numpy as np

from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta, ButcherTableau, BackwardEuler, DIRK43_2, EDIRK4, ESDIRK53, SDIRK3


class Jacobian:
    "Base class for quadratic Jacobian."
    def __init__(self, k, func, rdiff=1e-10):
        self.k = k
        self.func = func
        self.rdiff = rdiff

        self.jacobian = np.zeros((self.k, self.k))

    def eval(self, u, t):
        e = np.zeros(self.k)
        e[0] = 1
        for i in range(self.k):
            self.jacobian[:, i] = 1 / self.rdiff * (self.func(u + self.rdiff * e, t) - self.func(u, t))
            e = np.roll(e, 1)

    def inv(self):
        return np.linalg.inv(self.jacobian)


class RungeKuttaConstrained(RungeKutta):
    def __init__(self, params):
        super().__init__(params)

    def get_full_f(self, f):
        if type(f).__name__ in ['mesh', 'MeshDAE']:
            return f
        elif f is None:
            prob = self.level.prob
            return self.get_full_f(prob.dtype_f(prob.init, val=0))
        else:
            raise NotImplementedError(f'Type \"{type(f)}\" not implemented in Runge-Kutta sweeper')

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(prob.dtype_u(prob.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1].diff[:] += lvl.dt * self.coll.Qmat[m, j] * self.get_full_f(lvl.f[j]).diff[:]

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes.
        """

        lvl = self.level
        prob = lvl.prob

        # only if the level has been touched before
        assert lvl.status.unlocked

        # only if the level has been touched before
        assert lvl.status.unlocked
        assert lvl.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = prob.dtype_u(prob.init)
            rhs.diff[:] += lvl.u[0].diff[:]
            for j in range(1, m + 1):
                rhs.diff[:] += lvl.dt * self.QI[m + 1, j] * self.get_full_f(lvl.f[j]).diff[:]

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            if self.QI[m + 1, m + 1] != 0:
                lvl.u[m + 1][:] = prob.solve_system(
                    rhs, lvl.dt * self.QI[m + 1, m + 1], lvl.u[m], lvl.time + lvl.dt * self.coll.nodes[m + 1]
                )
            else:  # Explicit solve does not maybe make sense since DAEs should be solved implicitly
                lvl.u[m + 1][:] = rhs[:]

            # update function values (we don't usually need to evaluate the RHS at the solution of the step)
            if m < M - self.coll.num_solution_stages or self.params.eval_rhs_at_right_boundary:
                lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], lvl.time + lvl.dt * self.coll.nodes[m + 1])

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None


class BackwardEulerConstrained(RungeKuttaConstrained, BackwardEuler):
    pass


class DIRK43_2Constrained(RungeKuttaConstrained, DIRK43_2):
    pass


class EDIRK4Constrained(RungeKuttaConstrained, EDIRK4):
    pass


class DIRK(RungeKutta):
    """
    L-stable Diagonally Implicit RK method with four stages of order 3.
    Taken from [here](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
    """

    a = 0.4358665215

    nodes = np.array([a, -0.7, 0.8, 0.924556761814, 1.0])
    weights = np.array([0.896869652944, 0.0182725272734, -0.0845900310706, -0.266418670647, a])
    matrix = np.zeros((5, 5))
    matrix[0, 0] = a
    matrix[1, :2] = [-1.13586652150, a]
    matrix[2, :3] = [1.08543330679, -0.721299828287, a]
    matrix[3, :4] = [0.416349501547, 0.190984004184, -0.118643265417, a]
    matrix[4, :] = [0.896869652944, 0.0182725272734, -0.0845900310706, -0.266418670647, a]
    ButcherTableauClass = ButcherTableau

class DIRK5(RungeKutta):
    """DIRK of order 5 by Carpenter.
    """

    nodes = np.array([1.0 / 4.0, 3.0 / 4.0, 11.0 / 20.0, 1.0])
    weights = np.array([2.0 / 5.0, -3.0 / 5.0, 1.0 / 4.0, 1.0 / 4.0])
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 1.0 / 4.0
    matrix[1, :2] = [1.0 / 4.0, 1.0 / 4.0]
    matrix[2, :3] = [7.0 / 20.0, -1.0 / 20.0, 1.0 / 4.0]
    matrix[3, :] = weights
    ButcherTableauClass = ButcherTableau


class DIRK5_2(RungeKutta):
    """Another DIRK of order 5."""

    gamma = 0.25

    nodes = np.array([gamma, 0.5, 0.38773856, 0.61676893, 1.0])
    weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0])
    matrix = np.zeros((5, 5))
    matrix[0, 0] = gamma
    matrix[1, :2] = [1.0 / 4.0, gamma]
    matrix[2, :3] = [1.0 / 6.0, 1.0 / 6.0, gamma]
    matrix[3, :4] = [1.0 / 2.0, -1.0 / 4.0, 1.0 / 4.0, gamma]
    matrix[4, :] = weights  # [1.0 / 2.0, -3.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0, gamma]
    ButcherTableauClass = ButcherTableau


class DIRKConstrained(RungeKuttaConstrained, DIRK):
    pass

class DIRK5Constrained(RungeKuttaConstrained, DIRK5):
    pass

class DIRK5_2Constrained(RungeKuttaConstrained, DIRK5_2):
    pass

class ESDIRK53Constrained(RungeKuttaConstrained, ESDIRK53):
    pass

class SDIRK3Constrained(RungeKuttaConstrained, SDIRK3):
    pass
