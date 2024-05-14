import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class genericImplicitEmbedded(generic_implicit):
    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super().__init__(params)

        # get QI matrix
        self.QI = self.get_Qdelta_implicit(self.coll, qd_type=self.params.QI)

    def integrate(self):
        """
        Integrates the right-hand side

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
            for j in range(1, m + 1):
                me[-1].diff[:] += L.dt * self.coll.Qmat[m, j] * L.f[j].diff[:]

        return me
    
    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # update the MIN-SR-FLEX preconditioner
        if self.params.QI.startswith('MIN-SR-FLEX'):
            k = L.status.sweep
            if k > M:
                self.params.QI = "MIN-SR-S"
            else:
                self.params.QI = 'MIN-SR-FLEX' + str(k)
            self.QI = self.get_Qdelta_implicit(self.coll, qd_type=self.params.QI)

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # get -QdF(u^k)_m
            for j in range(1, M + 1):
                integral[m].diff[:] -= L.dt * self.QI[m + 1, j] * L.f[j].diff[:]

            # add initial value
            integral[m] += L.u[0].diff[:]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m].diff[:]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(P.init, val=0.0)
            rhs.diff[:] = integral[m].diff[:]
            for j in range(1, m + 1):
                rhs.diff[:] += L.dt * self.QI[m + 1, j] * L.f[j].diff[:]

            # implicit solve with prefactor stemming from the diagonal of Qd
            alpha = L.dt * self.QI[m + 1, m + 1]
            if alpha == 0:
                L.u[m + 1] = rhs
            else:
                L.u[m + 1] = P.solve_system(rhs, alpha, L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None
