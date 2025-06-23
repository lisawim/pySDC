from mpi4py import MPI

from pySDC.implementations.sweeper_classes.generic_implicit_MPI import SweeperMPI
from pySDC.playgrounds.DAE.genericImplicitDAE import genericImplicitConstrained, genericImplicitEmbedded


class genericImplicitConstrainedMPI(SweeperMPI, genericImplicitConstrained):

    def integrate(self, last_only=False):
        """
        Integrates the right-hand side. Here, only the differential variables are integrated.

        Parameters
        ----------
        last_only : bool, optional
            Integrate only the last node for the residual or all of them.

        Returns
        -------
        me : list of dtype_u
            Containing the integral as values.
        """

        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        for m in [self.coll.num_nodes - 1] if last_only else range(self.coll.num_nodes):
            integral = P.dtype_u(P.init, val=0.0)
            integral.diff[:] = L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1].diff[:]
            recvBuf = me if m == self.rank else None
            self.comm.Reduce(integral, recvBuf, root=m, op=MPI.SUM)

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes.
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        self.updateVariableCoeffs(L.status.sweep)

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        rhs = self.integrate()
        rhs.diff[:] -= L.dt * self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1].diff[:]
        rhs.diff[:] += L.u[0].diff[:]

        if L.tau[self.rank] is not None:
            rhs[:] += L.tau[self.rank].diff[:]

        # implicit solve with prefactor stemming from the diagonal of Qd
        alpha = L.dt * self.QI[self.rank + 1, self.rank + 1]
        L.u[self.rank + 1] = P.solve_system(
            rhs,
            alpha,
            L.u[self.rank + 1],
            L.time + L.dt * self.coll.nodes[self.rank],
        )

        # update function values
        L.f[self.rank + 1] = P.eval_f(L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q. For the residual, collocation matrix Q
        is only applied to the differential equations since no integration applies to the algebraic constraints.

        Parameters
        ----------
        stage : str
            The current stage of the step the level belongs to.
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # build QF(u)
        res = self.integrate(last_only=L.params.residual_type[:4] == 'last')
        res.diff[:] += L.u[0].diff[:] - L.u[self.rank + 1].diff[:]

        # add tau if associated
        if L.tau[self.rank] is not None:
            res.diff[:] += L.tau[self.rank].diff[:]

        # use abs function from data type here
        res_norm = abs(res)

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = self.comm.allreduce(res_norm, op=MPI.MAX)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = self.comm.bcast(res_norm, root=self.comm.size - 1)
        elif L.params.residual_type == 'full_rel':
            L.status.residual = self.comm.allreduce(res_norm / abs(L.u[0]), op=MPI.MAX)
        elif L.params.residual_type == 'last_rel':
            L.status.residual = self.comm.bcast(res_norm / abs(L.u[0]), root=self.comm.size - 1)
        else:
            raise NotImplementedError(f'residual type \"{L.params.residual_type}\" not implemented!')

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None


class genericImplicitEmbeddedMPI(SweeperMPI, genericImplicitEmbedded):

    def integrate(self, last_only=False):
        """
        Integrates the right-hand side.

        Parameters
        ----------
        last_only : bool, optional
            Integrate only the last node for the residual or all of them.

        Returns
        -------
        me : list of dtype_u
            Containing the integral as values.
        """

        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        for m in [self.coll.num_nodes - 1] if last_only else range(self.coll.num_nodes):
            recvBuf = me if m == self.rank else None
            self.comm.Reduce(
                L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1], recvBuf, root=m, op=MPI.SUM
            )

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes.
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        self.updateVariableCoeffs(L.status.sweep)

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        rhs = self.integrate()
        rhs[:] -= L.dt * self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1][:]
        rhs.diff[:] += L.u[0].diff[:]

        if L.tau[self.rank] is not None:
            rhs[:] += L.tau[self.rank].diff[:]

        # implicit solve with prefactor stemming from the diagonal of Qd
        alpha = L.dt * self.QI[self.rank + 1, self.rank + 1]
        if alpha == 0:
            L.u[self.rank + 1] = rhs
        else:
            L.u[self.rank + 1] = P.solve_system(
                rhs,
                alpha,
                L.u[self.rank + 1],
                L.time + L.dt * self.coll.nodes[self.rank],
            )

        # update function values
        L.f[self.rank + 1] = P.eval_f(L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_residual(self, stage=None):
        """
        Computation of the residual using the collocation matrix Q.

        Parameters
        ----------
        stage : str
            The current stage of the step the level belongs to.
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # build QF(u)
        res = self.integrate(last_only=L.params.residual_type[:4] == 'last')
        res.diff[:] += L.u[0].diff[:] - L.u[self.rank + 1].diff[:]

        # add tau if associated
        if L.tau[self.rank] is not None:
            res.diff[:] += L.tau[self.rank].diff[:]

        # use abs function from data type here
        res_norm = abs(res)

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = self.comm.allreduce(res_norm, op=MPI.MAX)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = self.comm.bcast(res_norm, root=self.comm.size - 1)
        elif L.params.residual_type == 'full_rel':
            L.status.residual = self.comm.allreduce(res_norm / abs(L.u[0]), op=MPI.MAX)
        elif L.params.residual_type == 'last_rel':
            L.status.residual = self.comm.bcast(res_norm / abs(L.u[0]), root=self.comm.size - 1)
        else:
            raise NotImplementedError(f'residual type \"{L.params.residual_type}\" not implemented!')

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
