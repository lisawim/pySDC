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

        params['skip_residual_computation'] = ()

        self.newton_tol = 1e-14
        self.newton_maxiter = 11

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes.
        """

        lvl = self.level
        prob = lvl.prob

        N = len(lvl.u[0].flatten())
        M = self.coll.num_nodes

        # Preallocate quantities
        u0_full = np.zeros((M, N))
        sys = np.zeros((M, N))
        f_init = np.zeros((M, N))

        # Initial guess for the stage values
        for m in range(M):
            u0_full[m] = lvl.u[0][:].flatten()
            f_init[m] = lvl.f[m + 1][:].flatten()

        du_new = prob.solve_collocation_system(
            self.F, f_init, lvl.time, lvl.dt, lvl, M, N, self.coll.Qmat, self, sys, u0_full
        )
        du_new = du_new.reshape(M, N)

        for m in range(M):
            # Reshape to datatype of dtype_f
            du_new_reshape = du_new[m].reshape(lvl.f[0].shape).view(type(lvl.f[0]))

            lvl.f[m + 1][:] = du_new_reshape

        # Update numerical solution
        integral = self.integrate()
        for m in range(M):
            lvl.u[m + 1][:] = lvl.u[0][:] + integral[m][:]

        self.du_init = prob.dtype_f(lvl.f[-1])

        lvl.status.updated = True

        return None

    def compute_residual(self, stage=None):
        r"""
        Uses the absolute value of the DAE system

        .. math::
            ||F(t, u, u')||

        for computing the residual in a chosen norm.

        Parameters
        ----------
        stage : str, optional
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

        # compute the residual for each node
        res_norm = []
        for m in range(self.coll.num_nodes):
            # use abs function from data type here
            res_norm.append(abs(P.eval_f(L.u[m + 1], L.f[m + 1], L.time + L.dt * self.coll.nodes[m])))

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

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None

    @staticmethod
    def F(du, t, dt, L, M, N, P, Qmat, sweep, sys, u0_full):
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

        local_du_approx = du.reshape(M, N)

        local_u_approx = u0_full.copy()

        # Applying quadrature to local approximation of u
        for m in range(M):
            for j in range(M):
                local_u_approx[m] += dt * Qmat[m + 1, j + 1] * local_du_approx[j]

        for m in range(M):
            # Reshaping to get internal datatype
            local_u_approx_reshape = local_u_approx[m].reshape(L.u[0].shape).view(type(L.u[0]))
            local_du_approx_reshape = local_du_approx[m].reshape(L.f[0].shape).view(type(L.f[0]))

            # Get the system and do flattening
            tau_m = t + dt * sweep.coll.nodes[m + 1]
            f_eval = P.eval_f(local_u_approx_reshape, local_du_approx_reshape, tau_m)
            sys[m] = f_eval.flatten()

        return sys.flatten()


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
