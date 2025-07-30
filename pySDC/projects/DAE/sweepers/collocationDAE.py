import numpy as np
from scipy.optimize import root
from scipy.sparse.linalg import gmres

from pySDC.implementations.sweeper_classes.Runge_Kutta import ButcherTableau, ButcherTableauEmbedded
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

        def impl_sys(du_unknown):
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

            local_du_approx = du_unknown.reshape(M, N)

            local_u_approx = u0_full.copy()

            # Applying quadrature to local approximation of u
            for m in range(M):
                for j in range(M):
                    local_u_approx[m] += lvl.dt * self.coll.Qmat[m + 1, j + 1] * local_du_approx[j]

            for m in range(M):
                # Reshaping to get internal datatype
                local_u_approx_reshape = local_u_approx[m].reshape(lvl.u[0].shape).view(type(lvl.u[0]))
                local_du_approx_reshape = local_du_approx[m].reshape(lvl.f[0].shape).view(type(lvl.f[0]))

                # Get the system and do flattening
                tau_m = lvl.time + lvl.dt * self.coll.nodes[m + 1]
                f_eval = prob.eval_f(local_u_approx_reshape, local_du_approx_reshape, tau_m)
                sys[m] = f_eval.flatten()

            return sys.flatten()

        # def jac(u, dt_jac=1e-8):
        #     """
        #     Computes the Jacobian for a multivariate function.
        #     Taken from
        #     https://scicomp.stackexchange.com/questions/19652/can-an-approximated-jacobian-with-finite-differences-cause-instability-in-the-ne

        #     Parameters
        #     ----------
        #     U : dtype_f
        #         Vector for which the Jacobian is computed.
        #     """
        #     n_jac, m_jac = len(u), len(impl_sys(u))
        #     jac = np.zeros((n_jac, m_jac))
        #     e = np.zeros(n_jac)
        #     e[0] = 1
        #     for k in range(n_jac):
        #         jac[:, k] = (1 / (dt_jac)) * (impl_sys(u + dt_jac * e) - impl_sys(u - dt_jac * e))
        #         # jac[:, k] = (1 / (dt_jac)) * (impl_sys(U + dt_jac * e) - impl_sys(U))
        #         e = np.roll(e, 1)
        #     return jac
        
        # du = f_init.flatten()

        # n = 0
        # newton_maxiter = 100
        # while n < newton_maxiter:
        #     g = impl_sys(du)
        #     res = np.linalg.norm(g, np.inf)

        #     if res < self.newton_tol:
        #         break

        #     dg = jac(du)

        #     # dx = np.linalg.solve(dg, g)
        #     dx, _ = gmres(dg, g, x0=du, atol=1e-14, maxiter=1000)

        #     du -= dx

        #     n += 1

        # du_new = du.reshape(M, N)
        # Solve implicit system and reshape
        # du_new = root(impl_sys, f_init.flatten(), method="hybr", tol=1e-14)
        # du_new = du_new.x.reshape(M, N)

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
