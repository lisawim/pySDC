import numpy as np

from pySDC.core.errors import ParameterError
from pySDC.implementations.sweeper_classes.Runge_Kutta import ButcherTableau
from pySDC.projects.DAE.sweepers.rungeKuttaDAE import RungeKuttaDAE
from pySDC.implementations.sweeper_classes.Runge_Kutta import ButcherTableau

from qmat import Q_GENERATORS


class CollocationDAE(RungeKuttaDAE):
    def __init__(self, params):
        super().__init__(params)

        # Store number of nodes here since in parent class num_nodes will be overwritten
        self.M = params['num_nodes']

        self.newton_tol = 1e-14
        self.newton_maxiter = 11

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes.
        """

        lvl = self.level
        prob = lvl.prob

        M = self.coll.num_nodes

        # Initial guesses for the stage values
        u0_full = [lvl.u[0][:] for _ in range(M)]
        f_init = [lvl.f[m + 1].flatten() for m in range(M)]

        du_new = prob.solve_collocation_system(
            self.F, f_init, lvl.time, lvl.dt, M, self.coll.Qmat, self, u0_full
        )

        for m in range(M):
            lvl.f[m + 1][:] = du_new[m]

        # Update numerical solution
        integral = self.integrate()
        for m in range(M):
            lvl.u[m + 1][:] = lvl.u[0][:] + integral[m][:]

        self.du_init = prob.dtype_f(lvl.f[-1])

        lvl.status.updated = True

        return None

    @staticmethod
    def F(du, t, dt, M, P, Qmat, sweep, u0_full):
        r"""
        This function builds the implicit system to be solved for a DAE of the form

        .. math::
            0 = F(u, u', t)

        Applying a collocation method yields the (non)-linear system to be solved

        .. math::
            0 = F(u_0 + \sum_{j=1}^M \tilde{q}_{mj} U_j, U_m, \tau_m),

        which is solved for the derivative of u.

        Note
        ----
        This function is differs from the implicit system function defined in ``fullyImplicitDAE``.
        The system is not solved node-by-node, but for all nodes simultaneously, since we have a
        dense coefficient matrix in the Butcher tableau.

        Parameters
        ----------
        du_unknown : np.1darray
            Unknowns of the system (derivative of solution u).

        Returns
        -------
        sys : np.1darray
            System to be solved.
        """

        local_du_approx = [du[m].copy() for m in range(M)]

        local_u_approx = [u0_full[m].copy() for m in range(M)]

        # Applying quadrature to local approximation of u
        for m in range(M):
            for j in range(M):
                local_u_approx[m] += dt * Qmat[m + 1, j + 1] * local_du_approx[j]

        sys = []
        for m in range(M):
            # Get the system and do flattening
            tau_m = t + dt * sweep.coll.nodes[m + 1]
            f_eval = P.eval_f(local_u_approx[m], local_du_approx[m], tau_m)
            sys.append(P.dtype_f(f_eval))

        sys_flatten = np.concatenate([sys_val.flatten() for sys_val in sys])
        return sys_flatten


class RadauIIA5DAE(CollocationDAE):
    """Method of Radau IIa family of order 5."""
    generator = Q_GENERATORS["Collocation"](
        nNodes=3, nodeType="LEGENDRE", quadType="RADAU-RIGHT", tLeft=0, tRight=1
    )

    nodes = generator.nodes.copy()
    weights = generator.weights.copy()
    matrix = generator.Q
    ButcherTableauClass = ButcherTableau


class RadauIIA7DAE(CollocationDAE):
    """Method of Radau IIa family of order 7."""
    generator = Q_GENERATORS["Collocation"](
        nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT", tLeft=0, tRight=1
    )

    nodes = generator.nodes.copy()
    weights = generator.weights.copy()
    matrix = generator.Q
    ButcherTableauClass = ButcherTableau


class RadauIIA9DAE(CollocationDAE):
    """Method of Radau IIa family of order 9."""
    generator = Q_GENERATORS["Collocation"](
        nNodes=5, nodeType="LEGENDRE", quadType="RADAU-RIGHT", tLeft=0, tRight=1
    )

    nodes = generator.nodes.copy()
    weights = generator.weights.copy()
    matrix = generator.Q
    ButcherTableauClass = ButcherTableau
