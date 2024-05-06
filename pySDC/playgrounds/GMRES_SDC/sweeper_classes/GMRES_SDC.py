import numpy as np
import scipy as sp
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.GMRES_SDC.helpers.Jacobian import Jacobian


def non_f(eps, t):
    return 1 / eps * np.cos(t) - np.sin(t)


class GMRES_SDC(generic_implicit):
    r"""
    Class implementing GMRES-SDC as proposed in [1]_.

    References
    ----------
    .. [1] Accelerating the convergence of spectral deferred correction methods.
        Jingfang Huang, Jun Jia and Michael L. Minion. J. Comput. Phys. 214, No. 2, 633-656 (2006)
    """

    def __init__(self, params):
        """Initialization routine for the custom sweeper"""

        if 'QI' not in params:
            params['QI'] = 'IE'

        if 'k0' not in params:
            params['k0'] = params['num_nodes']

        # call parent's initialization routine
        super().__init__(params)

        # get QI matrix
        self.QI = self.get_Qdelta_implicit(self.coll, qd_type=self.params.QI)
        self.pr_norm = []
        self.callback_pr_norm = lambda res: self.pr_norm.append(res)

    def update_nodes(self):
        """
        Updates values for u and f. This sweeper is only defined for Prothero-Robinson,
        and thus needs to be still updated for the general case!

        Returns
        -------
        None
        """

        lvl = self.level
        prob = lvl.prob

        # only if the level has been touched before
        assert lvl.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes
        nodes = [lvl.time + lvl.dt * self.coll.nodes[m] for m in range(M)]

        # vector with spreaded initial condition and nonhomogeneous part
        u0full = np.array([lvl.u[0].flatten() for m in range(M)]).flatten()
        non_ffull = np.array([non_f(prob.epsilon, tau) for tau in nodes])

        # define right-hand side, left-hand side and preconditioner of system
        A = np.eye(M) - lvl.dt * (- 1 / prob.epsilon) * self.coll.Qmat[1:, 1:]
        b = u0full + lvl.dt * self.coll.Qmat[1:, 1:].dot(non_ffull)

        # define preconditioner
        P = np.eye(M) - lvl.dt * (- 1 / prob.epsilon) * self.QI[1:, 1:]
        M_b = lambda b: spla.spsolve(sp.sparse.csr_matrix(P), b)
        Mpre = spla.LinearOperator((M, M), M_b)

        if self.params.k0 > 0:
            sol = gmres(
                A,
                b.flatten(),
                x0=u0full.flatten(),
                M=Mpre,
                restart=self.params.k0,
                atol=1e-14,
                rtol=0,
                maxiter=2,
                callback=self.callback_pr_norm,
                callback_type='pr_norm',
            )[0]
        else:
            # sol = (np.eye(M) - LHS).dot(ufull) + RHS
            raise NotImplementedError()

        n = len(lvl.u[0])
        for m in range(M):
            unew = prob.dtype_u(prob.init)
            unew[:] = sol[m * n : m * n + n]
            lvl.u[m + 1][:] = unew
            lvl.f[m + 1][:] = prob.eval_f(lvl.u[m + 1], nodes[m])

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None
