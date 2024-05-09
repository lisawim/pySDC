import numpy as np
np.set_printoptions(precision=30)

from pySDC.implementations.sweeper_classes.Runge_Kutta import (
    RungeKutta,
    ButcherTableau,
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

    def integrate(self):
        """
        Integrates the right-hand side.

        Returns
        -------
        me : list of dtype_u
            Containing the integral as values.
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        me = []
        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(prob.dtype_u(prob.init, val=0.0))
            for j in range(1, m + 1):
                me[-1][:] += lvl.dt * self.coll.Qmat[m, j] * lvl.f[j][:]

        return me

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


class SemiImplicitRungeKuttaDAE(RungeKuttaDAE):
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
                u_approx.diff[:] += lvl.dt * self.QI[m + 1, j] * lvl.f[j].diff[:]
            
            def F(du):
                """
                Build implicit system to solve in order to find the unknowns.

                Parameters
                ----------
                unknowns : dtype_u
                    Unknowns of the system.

                Returns
                -------
                sys :
                    System to be solved as implicit function.
                """

                du_mesh = prob.dtype_f(du)

                local_u_approx = prob.dtype_u(u_approx)

                local_u_approx.diff[:] += lvl.dt * self.QI[m + 1, m + 1] * du_mesh.diff[:]
                local_u_approx.alg[:] = du_mesh.alg[:]

                sys = prob.eval_f(local_u_approx, du_mesh, lvl.time + lvl.dt * self.coll.nodes[m + 1])
                return sys
            
            u0 = prob.dtype_u(prob.init)
            u0.diff[:], u0.alg[:] = lvl.f[m].diff[:], lvl.u[m].alg[:]
            u_new = prob.solve_system(F, u0[:], lvl.time + lvl.dt * self.coll.nodes[m + 1])

            lvl.f[m + 1].diff[:] = u_new.diff[:]
            lvl.u[m + 1].alg[:] = u_new.alg[:]

        # Update solution approximation - update only value at last node
        lvl.u[-1][:].diff[:] = lvl.u[0].diff[:]
        for j in range(1, M + 1):
            lvl.u[-1].diff[:] += lvl.dt * self.coll.Qmat[-1, j] * lvl.f[j].diff[:]

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None
            
        


class TrapezoidalRule(RungeKutta):
    """
    Famous trapezoidal rule of second order. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), third one in eq. (213).
    """

    nodes = np.array([0.0, 1.0])
    weights = np.array([1.0 / 2.0, 1.0 / 2.0])
    matrix = np.zeros((2, 2))
    matrix[0, 0] = 0.0
    matrix[1, :] = [1.0 / 2.0, 1.0 / 2.0]
    ButcherTableauClass = ButcherTableau


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

# semi-implicit RK variants

class SemiImplicitBackwardEulerDAE(SemiImplicitRungeKuttaDAE, BackwardEuler):
    pass


class SemiImplicitTrapezoidalRuleDAE(SemiImplicitRungeKuttaDAE, CrankNicholson):
    pass


class SemiImplicitKurdiEDIRK45_2DAE(SemiImplicitRungeKuttaDAE, KurdiEDIRK45_2):
    """
    For fully-implicit DAEs of index 2 the order of the
    scheme is only two (observation).
    """
    pass


class SemiImplicitEDIRK4DAE(SemiImplicitRungeKuttaDAE, EDIRK4):
    """
    For fully-implicit DAEs of index 2 the order of the
    scheme is only two (observation).
    """
    pass


class SemiImplicitDIRK43_2DAE(SemiImplicitRungeKuttaDAE, DIRK43_2):
    pass