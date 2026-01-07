import numpy as np

from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta, ButcherTableauEmbedded


class RungeKuttaAllowingExplicitSolve(RungeKutta):
    def get_full_f(self, f):
        """
        Get the full right hand side as a `mesh` from the right hand side

        Args:
            f (dtype_f): Right hand side at a single node

        Returns:
            mesh: Full right hand side as a mesh
        """
        if type(f).__name__ in ['mesh', 'MeshDAE']:
            return f
        elif type(f).__name__.lower() in ['imex_mesh']:
            return f.impl + f.expl
        elif f is None:
            prob = self.level.prob
            return self.get_full_f(prob.dtype_f(prob.init, val=0))
        else:
            raise NotImplementedError(f'Type \"{type(f)}\" not implemented in Runge-Kutta sweeper')

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes

        Returns:
            None
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        # only if the level has been touched before
        assert lvl.status.unlocked
        assert lvl.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = prob.dtype_u(lvl.u[0])
            for j in range(1, m + 1):
                rhs += lvl.dt * self.QI[m + 1, j] * self.get_full_f(lvl.f[j])

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            lvl.u[m + 1] = prob.solve_system(
                rhs, lvl.dt * self.QI[m + 1, m + 1], lvl.u[m], lvl.time + lvl.dt * self.coll.nodes[m + 1]
            )

            # update function values (we don't usually need to evaluate the RHS at the solution of the step)
            lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], lvl.time + lvl.dt * self.coll.nodes[m + 1])

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None


class DOPRI5(RungeKuttaAllowingExplicitSolve):
    """Method of Dormand & Prince of order 5."""

    ButcherTableauClass = ButcherTableauEmbedded

    nodes = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=float)

    b = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0], dtype=float)
    b2 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40], dtype=float)
    weights = np.vstack((b, b2))

    matrix = np.zeros((7, 7), dtype=float)
    matrix[1, 0] = 1 / 5

    matrix[2, 0] = 3 / 40
    matrix[2, 1] = 9 / 40

    matrix[3, 0] = 44 / 45
    matrix[3, 1] = -56 / 15
    matrix[3, 2] = 32 / 9

    matrix[4, 0] = 19372 / 6561
    matrix[4, 1] = -25360 / 2187
    matrix[4, 2] = 64448 / 6561
    matrix[4, 3] = -212 / 729

    matrix[5, 0] = 9017 / 3168
    matrix[5, 1] = -355 / 33
    matrix[5, 2] = 46732 / 5247
    matrix[5, 3] = 49 / 176
    matrix[5, 4] = -5103 / 18656

    matrix[6, 0] = 35 / 384
    matrix[6, 2] = 500 / 1113
    matrix[6, 3] = 125 / 192
    matrix[6, 4] = -2187 / 6784
    matrix[6, 5] = 11 / 84
