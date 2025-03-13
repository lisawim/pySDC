import numpy as np
import scipy.sparse.linalg as spla

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class GMRESSDC(generic_implicit):
    def integrate(self):
        """
        Overwrites the parent method! Integrates only the nonhomogeneous part of the
        right-hand side.

        Returns:
            list of dtype_u: containing the integral as values
        """

        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * P.eval_nonhomogeneous_part(L.time + L.dt * self.coll.nodes[j - 1])

        return me

    def update_nodes(self):
        r"""
        Updates values of ``u`` and ``f`` at collocation nodes. This correspond to a single iteration of the
        preconditioned Richardson iteration in **"ordinary"** SDC.
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        N = len(L.u[0].flatten())
        M = self.coll.num_nodes

        # get integral of nonhomogeneous part of right-hand side
        integral = self.integrate()
        for m in range(1, M + 1):
            integral[m - 1] += L.u[m]  # L.u[0]

        A = P.A

        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs += L.dt * self.QI[m + 1, j] * A.dot(L.u[j])

            sol, exit_code = spla.gmres(np.identity(N) - L.dt * self.QI[m + 1, m + 1] * A, rhs)

            L.u[m + 1][:] = sol

        L.status.updated = True

        return None
