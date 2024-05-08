import numpy as np

from pySDC.core.Problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae


class LinearTestDAE(ptype_dae):
    """
    Semi-explicit linear DAE of index one. It reads

    .. math::
        \dfrac{d}{dt}u_d = \lambda_d u_d + \lambda_a u_a,

    .. math::
        0 = \lambda_d u_d - \lambda_a u_a,

    where :math:`u_d` is the differential variable and :math:`u_a` is denoted as the algebraic
    variable. :math:`\lambda_d` and :math:`\lambda_a` are non-zero fixed parameters.

    Parameters
    ----------
    lamb_diff : float
        Parameter :math:`\lambda_d`.
    lamb_alg : float
        Parameter :math:`\lambda_a`.
    newton_tol : float
        Tolerance for inner solver to terminate.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts work, here, the number of right-hand side evaluations and work in inner solver
        are counted.
    """

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=(1, 1), newton_tol=newton_tol)
        self._makeAttributeAndRegister('newton_tol', localVars=locals())
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['newton'] = WorkCounter()

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        u_diff, u_alg = u.diff, u.alg
        du_diff = du.diff

        f = self.dtype_f(self.init)
        f.diff[:] = du_diff - 1.0 * u_diff - 1.0 * u_alg
        f.alg[:] = 1.0 * u_diff - 1.0 * u_alg
        return f

    def u_exact(self, t, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[:] = np.exp(2 * t)
        me.alg[:] = np.exp(2 * t)
        return me
