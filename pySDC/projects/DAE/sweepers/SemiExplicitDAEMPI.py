from mpi4py import MPI

from pySDC.projects.DAE.sweepers.fully_implicit_DAE_MPI import SweeperDAEMPI
from pySDC.projects.DAE.sweepers.SemiExplicitDAE import SemiExplicitDAE


class SemiExplicitDAEMPI(SweeperDAEMPI, SemiExplicitDAE):
    """
    TODO: Write detailed docu!
    """

    def integrate(self, last_only=False):
        """
        Integrates the gradient. Note that here only the differential part is integrated, i.e., the
        integral over the algebraic part is zero. ``me`` serves as buffer, and root process ``m``
        stores the result of integral at node :math:`\tau_m` there.

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
            recvBufDiff =me.diff[:] if m == self.rank else None
            recvBufAlg = me.alg[:] if m == self.rank else None
            integral = P.dtype_u(P.init, val=0.0)
            integral.diff[:] = L.dt * self.coll.Qmat[m + 1, self.rank + 1] * L.f[self.rank + 1].diff
            self.comm.Reduce(
                integral.diff, recvBufDiff, root=m, op=MPI.SUM
            )
            self.comm.Reduce(
                integral.alg, recvBufAlg, root=m, op=MPI.SUM
            )

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

        integral = self.integrate()
        integral.diff -= L.dt * self.QI[self.rank + 1, self.rank + 1] * L.f[self.rank + 1].diff
        integral.diff += L.u[0].diff

        u_approx = P.dtype_u(integral)
        def implSystem(unknowns):
            """
            Build implicit system to solve in order to find the unknowns.

            Parameters
            ----------
            params : dtype_u
                Unknowns of the system.

            Returns
            -------
            sys :
                System to be solved as implicit function.
            """
            unknowns_mesh = P.dtype_f(unknowns)

            # add implicit factor with unknown
            local_u_approx = P.dtype_f(u_approx)
            local_u_approx.diff += L.dt * self.QI[self.rank + 1, self.rank + 1] * unknowns_mesh.diff
            local_u_approx.alg = unknowns_mesh.alg

            sys = P.eval_f(local_u_approx, unknowns_mesh, L.time + L.dt * self.coll.nodes[self.rank])
            return sys

        u0 = P.dtype_u(P.init)
        u0.diff[:], u0.alg[:] = L.f[self.rank + 1].diff, L.u[self.rank + 1].alg
        u_new = P.solve_system(
            implSystem, u0, L.time + L.dt * self.coll.nodes[self.rank]
        )
        L.f[self.rank + 1].diff = u_new.diff
        L.u[self.rank + 1].alg = u_new.alg

        integral = self.integrate()
        L.u[self.rank + 1].diff = L.u[0].diff + integral.diff

        # indicate presence of new values at this level
        L.status.updated = True

        return None
