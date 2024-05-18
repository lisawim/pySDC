import numpy as np
np.printoptions(precision=30)

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.core.Errors import ParameterError


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
            for j in range(1, self.coll.num_nodes + 1):
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
            integral[m].diff[:] += L.u[0].diff[:]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m][:] += L.tau[m].diff[:]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs.diff[:] += L.dt * self.QI[m + 1, j] * L.f[j].diff[:]

            # implicit solve with prefactor stemming from the diagonal of Qd
            alpha = L.dt * self.QI[m + 1, m + 1]
            if alpha == 0:
                L.u[m + 1][:] = rhs[:]
            else:
                L.u[m + 1][:] = P.solve_system(rhs, alpha, L.u[m + 1][:], L.time + L.dt * self.coll.nodes[m])
            # update function values
            L.f[m + 1][:] = P.eval_f(L.u[m + 1][:], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_residual(self, stage=''):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            res[m].diff[:] += L.u[0].diff[:] - L.u[m + 1].diff[:]
            # add tau if associated
            if L.tau[m] is not None:
                res[m].diff[:] += L.tau[m].diff[:]
            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = max(res_norm)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = res_norm[-1]
        elif L.params.residual_type == 'full_rel':
            L.status.residual = max(res_norm) / abs(L.u[0])
        elif L.params.residual_type == 'last_rel':
            L.status.residual = res_norm[-1] / abs(L.u[0])
        else:
            raise ParameterError(
                f'residual_type = {L.params.residual_type} not implemented, choose '
                f'full_abs, last_abs, full_rel or last_rel instead'
            )
        # print(L.dt, L.status.residual)
        # indicate that the residual has seen the new values
        L.status.updated = False

        return None


class genericImplicitEmbedded2(generic_implicit):
    r"""
    This is a test sweeper for solving differential-algebraic equations of the form

    .. math::
        y' = f(y, z),

    .. math::
        0 = g(y, z).
    
    When the :math:`\varepsilon`-embedding is applied to the SDC scheme for a singular
    perturbed problem we end up with the scheme

    .. math::
       \mathbf{y}^{k+1} = \mathbf{y}_0 + \Delta t (\mathbf{Q} - \mathbf{Q}_\Delta)\otimes \mathbf{I}_{N_d}f(\mathbf{y}^{k}, \mathbf{z}^{k}) +
       \Delta t \mathbf{Q}_\Delta\otimes \mathbf{I}_{N_d}f(\mathbf{y}^{k+1}, \mathbf{z}^{k+1}),

    .. math::
       \mathbf{0} = \Delta t (\mathbf{Q} - \mathbf{Q}_\Delta)\otimes \mathbf{I}_{N_d}g(\mathbf{y}^{k}, \mathbf{z}^{k}) +
       \Delta t \mathbf{Q}_\Delta\otimes \mathbf{I}_{N_a}g(\mathbf{y}^{k+1}, \mathbf{z}^{k+1}).
    """
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
                integral[m][:] -= L.dt * self.QI[m + 1, j] * L.f[j][:]

            # add initial value - here, u0 is only added to differential part
            integral[m].diff[:] += L.u[0].diff[:]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m][:] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs[:] += L.dt * self.QI[m + 1, j] * L.f[j][:]

            # implicit solve with prefactor stemming from the diagonal of Qd
            alpha = L.dt * self.QI[m + 1, m + 1]
            if alpha == 0:
                L.u[m + 1][:] = rhs[:]
            else:
                L.u[m + 1][:] = P.solve_system(rhs, alpha, L.u[m + 1][:], L.time + L.dt * self.coll.nodes[m])
            # update function values
            L.f[m + 1][:] = P.eval_f(L.u[m + 1][:], L.time + L.dt * self.coll.nodes[m])
        print()
        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_residual(self, stage=''):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            res[m].diff[:] += L.u[0].diff[:] - L.u[m + 1].diff[:]
            # add tau if associated
            if L.tau[m] is not None:
                res[m][:] += L.tau[m][:]
            # use abs function from data type here
            res_norm.append(abs(res[m]))

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = max(res_norm)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = res_norm[-1]
        elif L.params.residual_type == 'full_rel':
            L.status.residual = max(res_norm) / abs(L.u[0])
        elif L.params.residual_type == 'last_rel':
            L.status.residual = res_norm[-1] / abs(L.u[0])
        else:
            raise ParameterError(
                f'residual_type = {L.params.residual_type} not implemented, choose '
                f'full_abs, last_abs, full_rel or last_rel instead'
            )
        # print(L.dt, L.status.residual)
        # indicate that the residual has seen the new values
        L.status.updated = False

        return None