import numpy as np
from mpi4py import MPI

from pySDC.core.sweeper import Sweeper, ParameterError
from pySDC.implementations.sweeper_classes.generic_implicit_MPI import SweeperMPI
from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained
import logging


class SweeperMPIGrouped(Sweeper):
    def __init__(self, params, level):
        self.logger = logging.getLogger('sweeper')

        if 'comm' not in params.keys():
            params['comm'] = MPI.COMM_WORLD
            self.logger.debug('Using MPI.COMM_WORLD for the communicator because none was supplied in the params.')
        super().__init__(params, level)

        self.M = self.coll.num_nodes
        self.P = self.params.comm.size
        if self.P > self.M:
            raise NotImplementedError(f"Need comm.size <= num_nodes. Got {self.P=} and {self.M=}.")

        self.local_nodes = [m for m in range(self.M) if (m % self.P) == self.params.comm.rank]

    @property
    def comm(self):
        return self.params.comm

    @property
    def rank(self):
        return self.comm.rank

    def owner(self, m: int) -> int:
        """Rank that 'owns' node m."""
        return m % self.P

    def compute_end_point(self):
        L = self.level
        P = L.prob

        if self.coll.right_is_node and not self.params.do_coll_update:
            last_node = self.M - 1
            root = self.owner(last_node)

            L.uend = P.dtype_u(L.u[last_node + 1]) if self.rank == root else P.dtype_u(L.u[0])
            self.comm.Bcast(L.uend, root=root)
        else:
            raise NotImplementedError("require last node to be identical with right interval boundary")

        return None

    def compute_residual(self, stage=None):
        L = self.level

        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # local maximum residual over nodes owned by this rank
        local_max = 0.0

        for m in self.local_nodes:
            # Node-wise integration
            res = self.integrate_node(m, last_only=False)
            res += L.u[0] - L.u[m + 1]
            if L.tau[m] is not None:
                res += L.tau[m]
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
                res_last = self.integrate_node(last_node, last_only=True)
                res_last += L.u[0] - L.u[last_node + 1]
                if L.tau[last_node] is not None:
                    res_last += L.tau[last_node]
                send_val = abs(res_last)
                if L.params.residual_type == "last_rel":
                    send_val = send_val / abs(L.u[0])
            L.status.residual = self.comm.bcast(send_val, root=root)
        else:
            raise NotImplementedError(f'residual type "{L.params.residual_type}" not implemented!')

        L.status.updated = False

        return None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)

        for m in self.local_nodes:
            if self.params.initial_guess == 'spread':
                # copy u[0] to all collocation nodes, evaluate RHS
                L.u[m + 1] = P.dtype_u(L.u[0])
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
            elif self.params.initial_guess == 'copy':
                # copy u[0] and RHS evaluation to all collocation nodes
                L.u[m + 1] = P.dtype_u(L.u[0])
                L.f[m + 1] = P.dtype_f(L.f[0])
            elif self.params.initial_guess == 'zero':
                # zeros solution for u and RHS
                L.u[m + 1] = P.dtype_u(init=P.init, val=0.0)
                L.f[m + 1] = P.dtype_f(init=P.init, val=0.0)
            else:
                raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        self._sync_u(sync_diff=True, sync_alg=True)
        self._sync_f()

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def _sync_f(self):
        """
        Synchronize stage derivatives L.f[m+1].diff across all ranks.

        Uses Allreduce(SUM) with the assumption that each stage m is written by its unique owner rank.
        """
        L = self.level
        P = L.prob
        M = self.M

        # Some sweepers/problems may not use f at all stages, but for DAEs it is typically present
        nd = L.f[0].diff.size
        dtype = L.f[0].diff.dtype

        # print(f"[rank {self.rank}] ENTER _sync_f", flush=True)

        local = np.zeros((M, nd), dtype=dtype)
        for m in self.local_nodes:
            local[m, :] = L.f[m + 1].diff[:]

        glob = np.zeros_like(local)
        self.comm.Allreduce(local, glob, op=MPI.SUM)

        for m in range(M):
            if L.f[m + 1] is None:
                L.f[m + 1] = P.dtype_f(P.init, val=0.0)
            L.f[m + 1].diff[:] = glob[m, :]

        # print(f"[rank {self.rank}] EXIT  _sync_f", flush=True)

    def _sync_u(self, sync_diff=True, sync_alg=True):
        """
        Synchronize stage values L.u[m+1] across all ranks.

        Parameters
        ----------
        sync_diff : bool
            Synchronize differential part u.diff
        sync_alg : bool
            Synchronize algebraic part u.alg
        """
        L = self.level
        P = L.prob
        M = self.M

        # print(f"[rank {self.rank}] ENTER _sync_u(diff={sync_diff}, alg={sync_alg})", flush=True)

        if sync_diff:
            nd = L.u[0].diff.size
            dtype = L.u[0].diff.dtype

            local = np.zeros((M, nd), dtype=dtype)
            for m in self.local_nodes:
                local[m, :] = L.u[m + 1].diff[:]

            glob = np.zeros_like(local)
            self.comm.Allreduce(local, glob, op=MPI.SUM)

            for m in range(M):
                if L.u[m + 1] is None:
                    L.u[m + 1] = P.dtype_u(P.init, val=0.0)
                L.u[m + 1].diff[:] = glob[m, :]

        if sync_alg and hasattr(L.u[0], "alg"):
            na = L.u[0].alg.size
            dtype_alg = L.u[0].alg.dtype

            local_alg = np.zeros((M, na), dtype=dtype_alg)
            for m in self.local_nodes:
                local_alg[m, :] = L.u[m + 1].alg[:]

            glob_alg = np.zeros_like(local_alg)
            self.comm.Allreduce(local_alg, glob_alg, op=MPI.SUM)

            for m in range(M):
                L.u[m + 1].alg[:] = glob_alg[m, :]

        # print(f"[rank {self.rank}] EXIT _sync_u", flush=True)


class genericImplicitConstrainedMPIGrouped(SweeperMPIGrouped, genericImplicitConstrained):
    """
    MPI-based collocation-parallel sweeper with grouped collocation nodes per rank.
    In contrast to genericImplicitConstrainedMPI, multiple collocation nodes may be
    processed sequentially on a single MPI rank.
    """

    def __init__(self, params, level):
        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super().__init__(params, level)

        # get QI matrix
        self.QI = self.get_Qdelta_implicit(qd_type=self.params.QI)

    def _ndiff(self):
        return self.level.u[0].diff.size

    def integrate(self, last_only=False):
        """
        Returns an array I of shape (M, ndiff) with
            I[m] = dt * sum_j Q[m+1, j+1] * f[j+1].diff
        If last_only, only fills m = M-1 (others zero).
        """
        L = self.level
        nd = self._ndiff()
        M = self.M

        # print(f"[rank {self.rank}] ENTER integrate(last_only={last_only})", flush=True)

        dtype = L.f[0].diff.dtype
        local_I = np.zeros((M, nd), dtype=dtype)

        target_ms = [M - 1] if last_only else range(M)

        # local contributions from owned source nodes j
        for j in self.local_nodes:
            fj = L.f[j + 1].diff
            for m in target_ms:
                local_I[m, :] += L.dt * self.coll.Qmat[m + 1, j + 1] * fj

        # Sum contributions across ranks
        I = np.zeros_like(local_I)
        self.comm.Allreduce(local_I, I, op=MPI.SUM)
        # print(f"[rank {self.rank}] EXIT integrate", flush=True)
        return I

    def integrate_node(self, m, last_only=False):
        """
        Convenience: return dtype_u integral for node m.
        """
        L = self.level
        P = L.prob
        I = self.integrate(last_only=last_only)
        me = P.dtype_u(P.init, val=0.0)
        me.diff[:] = I[m, :]
        return me

    def update_nodes(self):
        L = self.level
        P = L.prob
        assert L.status.unlocked

        # print(
        #     f"[rank {self.rank}] ENTER update_nodes: "
        #     f"time={L.time}, sweep={L.status.sweep}, iter={getattr(L.status,'iter',None)}",
        #     flush=True
        # )

        self.updateVariableCoeffs(L.status.sweep)

        # Build QF(u^k) for all m
        I = self.integrate(last_only=False)

        # Serial only along local nodes
        for m in self.local_nodes:
            # print(f"[rank {self.rank}] solve m={m}", flush=True)
            rhs = P.dtype_u(P.init, val=0.0)
            rhs.diff[:] = I[m, :]

            # Subtract QΔ contribution (diagonal of QI for node m)
            rhs.diff[:] -= L.dt * self.QI[m + 1, m + 1] * L.f[m + 1].diff[:]
            rhs.diff[:] += L.u[0].diff[:]

            if L.tau[m] is not None:
                rhs.diff[:] += L.tau[m].diff[:]

            alpha = L.dt * self.QI[m + 1, m + 1]
            L.u[m + 1] = P.solve_system(
                rhs,
                alpha,
                L.u[m + 1],
                L.time + L.dt * self.coll.nodes[m],
            )
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            # --- DEBUG: NaN/Inf checks (local nodes only) ---
            # if not np.isfinite(L.u[m + 1].diff[:]).all():
            #     raise RuntimeError(f"[rank {self.rank}] NaN/Inf in u.diff at node m={m}, time={L.time}, sweep={L.status.sweep}")
            # if not np.isfinite(L.f[m + 1].diff[:]).all():
            #     raise RuntimeError(f"[rank {self.rank}] NaN/Inf in f.diff at node m={m}, time={L.time}, sweep={L.status.sweep}")

        # After local updates, other ranks need the updated u/f values
        # print(f"[rank {self.rank}] before _sync_f", flush=True)
        self._sync_f()
        # print(f"[rank {self.rank}] after  _sync_f", flush=True)

        # print(f"[rank {self.rank}] before _sync_u", flush=True)
        self._sync_u(sync_diff=True, sync_alg=False)
        # print(f"[rank {self.rank}] after  _sync_u", flush=True)

        L.status.updated = True
        # --- DEBUG CONSISTENCY CHECK (4) ---
        val = int(L.status.updated)
        s = self.comm.allreduce(val, op=MPI.SUM)
        # print(
        #     f"[rank {self.rank}] DEBUG updated-flag sum={s} (should be {self.comm.size})",
        #     flush=True
        # )

        # print(f"[rank {self.rank}] EXIT  update_nodes", flush=True)
        return None
