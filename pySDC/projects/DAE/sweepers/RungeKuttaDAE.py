import numpy as np
np.set_printoptions(precision=30)

from pySDC.implementations.sweeper_classes.Runge_Kutta import (
    RungeKutta,
    BackwardEuler,
    CrankNicholson,
    KurdiEDIRK45_2,
    EDIRK4,
    DIRK43_2,
)


class RungeKuttaDAE(RungeKutta):
    def __init__(self, params):
        super().__init__(params)
        self.du_init = None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep.
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        du_init = self.du_init[:] if self.du_init is not None else prob.du_exact(lvl.time)[:]
        lvl.f[0] = prob.dtype_f(du_init)
        for m in range(1, self.coll.num_nodes + 1):
            lvl.u[m] = prob.dtype_u(init=prob.init, val=0.0)
            lvl.f[m] = prob.dtype_f(init=prob.init, val=0.0)

        lvl.status.unlocked = True
        lvl.status.updated = True

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        # get current level and problem description
        lvl = self.level
        prob = lvl.prob

        # only if the level has been touched before
        assert lvl.status.unlocked
        assert lvl.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        M = self.coll.num_nodes
        for m in range(M):
            u_approx = prob.dtype_u(lvl.u[0])
            for j in range(1, m + 1):
                u_approx += lvl.dt * self.QI[m + 1, j] * lvl.f[j][:]

            def F(du):
                r"""
                This function builds the implicit system to be solved for a DAE of the form

                .. math::
                    0 = F(u, u', t)

                Applying a RK method yields the (non)-linear system to be solved

                .. math::
                    0 = F(u_0 + \Delta t \sum_{j=1}^m a_{ij} U_j, U_m, \tau_m),

                which is solved for the derivative of u.

                Parameters
                ----------
                du : dtype_u
                    Unknowns of the system (derivative of solution u).

                Returns
                -------
                sys : dtype_f
                    System to be solved.
                """
                du_mesh = prob.dtype_f(du)

                local_u_approx = prob.dtype_u(u_approx)

                # defines the "implicit" factor, note that for explicit RK the diagonal element is zero
                local_u_approx += lvl.dt * self.QI[m + 1, m + 1] * du_mesh

                sys = prob.eval_f(local_u_approx, du_mesh, lvl.time + lvl.dt * self.coll.nodes[m + 1])
                return sys

            finit = prob.dtype_f(lvl.f[m])
            lvl.f[m + 1][:] = prob.solve_system(F, finit.flatten(), lvl.time + lvl.dt * self.coll.nodes[m + 1])

        # Update numerical solution - update value only at last node
        lvl.u[-1][:] = lvl.u[0]
        for j in range(1, M + 1):
            lvl.u[-1][:] += lvl.dt * self.coll.Qmat[-1, j] * lvl.f[j][:]

        self.du_init = prob.dtype_f(lvl.f[-1])

        lvl.status.updated = True

        return None


class SemiImplicitRungeKuttaDAE(RungeKutta):
    def integrate(self):
        r"""
        Returns the solution by integrating its gradient (fundamental theorem of calculus) at each collocation node.
        ``level.f`` stores the gradient of solution ``level.u``.

        Returns
        -------
        me : list of lists
            Integral of the gradient at each collocation node.
        """

        # get current level and problem description
        lvl = self.level
        prob = lvl.prob
        M = self.coll.num_nodes

        me = []
        for m in range(1, M + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(prob.dtype_u(prob.init, val=0.0))
            for j in range(1, M + 1):
                me[-1].diff[:] += lvl.dt * self.coll.Qmat[m, j] * lvl.f[j].diff[:]

        return me

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
            rhs = prob.dtype_u(prob.init, val=0.0)
            rhs.diff[:] = lvl.u[0].diff[:]
            for j in range(1, m + 1):
                rhs.diff[:] += lvl.dt * self.QI[m + 1, j] * lvl.f[j].diff[:]

            # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
            # if self.coll.implicit:
            #     lvl.u[m + 1][:] = prob.solve_system(
            #         rhs, lvl.dt * self.QI[m + 1, m + 1], lvl.u[m], lvl.time + lvl.dt * self.coll.nodes[m + 1]
            #     )
            # else:
            #     lvl.u[m + 1][:] = rhs[:]
            lvl.u[m + 1][:] = prob.solve_system(
                rhs, lvl.dt * self.QI[m + 1, m + 1], lvl.u[m], lvl.time + lvl.dt * self.coll.nodes[m + 1]
            )

            # update function values (we don't usually need to evaluate the RHS at the solution of the step)
            if m < M - self.coll.num_solution_stages or self.params.eval_rhs_at_right_boundary:
                lvl.f[m + 1] = prob.eval_f(lvl.u[m + 1], lvl.time + lvl.dt * self.coll.nodes[m + 1])

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None


class BackwardEulerDAE(RungeKuttaDAE, BackwardEuler):
    pass


class TrapezoidalRuleDAE(RungeKuttaDAE, CrankNicholson):
    pass


class KurdiEDIRK45_2DAE(RungeKuttaDAE, KurdiEDIRK45_2):
    """
    For fully-implicit DAEs of index 2 the order of the
    scheme is only two (observation).
    """
    pass


class EDIRK4DAE(RungeKuttaDAE, EDIRK4):
    """
    For fully-implicit DAEs of index 2 the order of the
    scheme is only two (observation).
    """
    pass


class DIRK43_2DAE(RungeKuttaDAE, DIRK43_2):
    pass


class SemiImplicitTrapezoidalRuleDAE(SemiImplicitRungeKuttaDAE, CrankNicholson):
    pass


class SemiImplicitEDIRK4DAE(SemiImplicitRungeKuttaDAE, EDIRK4):
    pass
