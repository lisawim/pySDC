import numpy as np
from mpi4py import MPI

from pySDC.core.errors import ParameterError
from pySDC.projects.DAE.sweepers.genericImplicitConstrainedMPIGrouped import SweeperMPIGrouped
from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE


class SweeperDAEMPIGrouped(SweeperMPIGrouped):

    def compute_residual(self, stage=None):
        r"""
        Uses the absolute value of the DAE system

        .. math::
            ||F(t, u, u')||

        for computing the residual in a chosen norm. If norm computes residual by using all values on collocation nodes, result is broadcasted to all processes; when norm only uses the value on last node, the result is collected on last process!

        Parameters
        ----------
        stage : str, optional
            The current stage of the step the level belongs to.
        """

        L = self.level
        P = L.prob

        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # local maximum residual over nodes owned by this rank
        local_max = 0.0
        for m in self.local_nodes:
            res = P.eval_f(L.u[m + 1], L.f[m + 1], L.time + L.dt * self.coll.nodes[m])
            local_max = max(local_max, abs(res))

        if L.params.residual_type in ("full_abs", "full_rel"):
            if L.params.residual_type == "full_rel":
                local_max = local_max / abs(L.u[0])
            L.status.residual = self.comm.allreduce(local_max, op=MPI.MAX)

        elif L.params.residual_type in ("last_abs", "last_rel"):
            last_node = self.M - 1
            root = self.owner(last_node)

            # Only owner of last node contributes meaningful value
            send_val = None
            if self.rank == root:
                res_last = P.eval_f(L.u[last_node + 1], L.f[last_node + 1], L.time + L.dt * self.coll.nodes[last_node])
                send_val = abs(res_last)
                if L.params.residual_type == "last_rel":
                    send_val = send_val / abs(L.u[0])
            L.status.residual = self.comm.bcast(send_val, root=root)
        else:
            raise NotImplementedError(f'residual type "{L.params.residual_type}" not implemented!')

        L.status.updated = False
        return None

    def predict(self):
        r"""
        Predictor to fill values at nodes before first sweep.

        For the fully implicit (FI) sweeper, ``level.f`` stores the solution derivative and is
        initially unknown. Therefore, we initialise ``level.f`` with zeros.

        In the grouped MPI variant, multiple collocation stages may be assigned to a single MPI rank.
        Hence, we initialise all stages owned by this rank and then synchronize if required.
        """

        L = self.level
        P = L.prob

        L.f[0] = P.dtype_f(init=P.init, val=0.0)

        for m in self.local_nodes:
            if self.params.initial_guess == "spread":
                L.u[m + 1] = P.dtype_u(L.u[0])
                L.f[m + 1] = P.dtype_f(init=P.init, val=0.0)
            elif self.params.initial_guess == "zero":
                L.u[m + 1] = P.dtype_u(init=P.init, val=0.0)
                L.f[m + 1] = P.dtype_f(init=P.init, val=0.0)
            else:
                raise ParameterError(f"initial_guess option {self.params.initial_guess} not implemented")

        self._sync_u(sync_diff=True, sync_alg=True)
        self._sync_f()

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True


class SemiImplicitDAEMPIGrouped(SweeperDAEMPIGrouped, SemiImplicitDAE):

    def _ndiff(self):
        """Number of differential unknowns."""
        return self.level.f[0].diff.size

    def _integrate_all_targets_diff(self, last_only=False):
        r"""
        Compute differential integrals for all target nodes m:

            I[m] = dt * sum_{j=0}^{M-1} Q[m+1, j+1] * f[j+1].diff

        Returns
        -------
        I : np.ndarray of shape (M, ndiff)
        """
        L = self.level
        M = self.M
        nd = self._ndiff()

        dtype = L.f[0].diff.dtype
        local_I = np.zeros((M, nd), dtype=dtype)

        targets = [M - 1] if last_only else range(M)

        for j in self.local_nodes:
            fj = L.f[j + 1].diff[:]
            for m in targets:
                local_I[m, :] += L.dt * self.coll.Qmat[m + 1, j + 1] * fj

        I = np.zeros_like(local_I)
        self.comm.Allreduce(local_I, I, op=MPI.SUM)
        return I

    def integrate(self, last_only=False):
        r"""
        Integrate the gradient (differential part only).

        In the grouped setting there is no longer a unique mapping rank <-> node.
        This routine therefore returns the integral for the *first local node* on this rank
        (or zeros if this rank owns no nodes). For node-wise usage in update_nodes(),
        prefer _integrate_all_targets_diff and index with m.
        """
        L = self.level
        P = L.prob

        me = P.dtype_u(P.init, val=0.0)
        I = self._integrate_all_targets_diff(last_only=last_only)

        if len(self.local_nodes) > 0:
            m0 = self.local_nodes[0]
            me.diff[:] = I[m0, :]
        return me

    def update_nodes(self):
        r"""
        Updates values of ``u`` and ``f`` at collocation nodes (single sweep / iteration).
        """
        L = self.level
        P = L.prob

        assert L.status.unlocked

        self._sync_f()

        # First integral based on current f
        I = self._integrate_all_targets_diff(last_only=False)

        # Local solves for all nodes owned by this rank
        for m in self.local_nodes:
            # Build u_approx = u0 + QF - QΔF (only diff part)  [as in your original code]
            integral_m = P.dtype_u(P.init, val=0.0)
            integral_m.diff[:] = I[m, :]

            integral_m.diff[:] -= L.dt * self.QI[m + 1, m + 1] * L.f[m + 1].diff[:]
            integral_m.diff[:] += L.u[0].diff[:]

            u_approx = P.dtype_u(integral_m)

            # u0 contains derivative guess (diff) and algebraic variables (alg)
            u0 = P.dtype_u(P.init)
            u0.diff[:] = L.f[m + 1].diff[:]
            u0.alg[:] = L.u[m + 1].alg[:]

            u_new = P.solve_system(
                SemiImplicitDAE.F,
                u_approx,
                L.dt * self.QI[m + 1, m + 1],
                u0,
                L.time + L.dt * self.coll.nodes[m],
            )

            # write back updated derivative and algebraic part
            L.f[m + 1].diff[:] = u_new.diff[:]
            L.u[m + 1].alg[:] = u_new.alg[:]

        self._sync_f()
        self._sync_u(sync_diff=False, sync_alg=True)

        # Second integral with updated f to update u.diff
        I2 = self._integrate_all_targets_diff(last_only=False)
        for m in self.local_nodes:
            L.u[m + 1].diff[:] = L.u[0].diff[:] + I2[m, :]

        # Optional but usually useful: make u.diff consistent globally
        self._sync_u(sync_diff=True, sync_alg=False)

        L.status.updated = True
        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval.

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        if not self.coll.right_is_node or self.params.do_coll_update:
            raise NotImplementedError()

        # SweeperMPIGrouped(self.params, self.level).compute_end_point()
        super().compute_end_point()
