import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class LinearTestDAE(ProblemDAE):
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

        self.lamb_diff = -2.0
        self.lamb_alg = 1.0

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
        f.diff[:] = du_diff - self.lamb_diff * u_diff - self.lamb_alg * u_alg
        f.alg[:] = self.algebraicConstraints(u, t)
        return f

    def algebraicConstraints(self, u, t):
        r"""
        Returns the algebraic constraints of the semi-explicit DAE system.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        f = self.dtype_f(self.init)
        f.alg[:] = self.lamb_diff * u.diff[:] - self.lamb_alg * u.alg[:]
        return f.alg

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
        me.diff[0] = np.exp(2 * self.lamb_diff * t)
        me.alg[0] = self.lamb_diff / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me


class LinearTestDAEConstrained(LinearTestDAE):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    """

    def __init__(self, nvars=(1, 1), lintol=1e-12, liniter=100):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('lintol', 'liniter', localVars=locals())
        self.work_counters['gmres'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.lamb_diff = -2.0
        self.lamb_alg = 1.0

        self.A = np.zeros((2, 2))
        self.A[0, :] = [self.lamb_diff, self.lamb_alg]
        self.A[1, :] = [self.lamb_diff, -self.lamb_alg]

        self.Aalg = np.zeros((2, 2))
        self.Aalg[1, :] = self.A[1, :]

        self.Id0 = sp.diags_array([1, 0], offsets=0)

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        u_diff, u_alg = u.diff, u.alg

        f = self.dtype_f(self.init)
        f.diff[:] = self.lamb_diff * u_diff + self.lamb_alg * u_alg
        f.alg[:] = self.algebraicConstraints(u, t)
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)
        u, info = gmres(
            A=self.Id0 - factor * self.A - self.Aalg,
            b=rhs.flatten(),
            x0=u0.flatten(),
            rtol=self.lintol,
            maxiter=self.liniter,
            atol=0,
            callback=self.work_counters['gmres'],
            callback_type='legacy',
        )
        me[:] = u.reshape(me.shape)
        return me


class LinearTestDAEEmbedded(LinearTestDAEConstrained):
    r"""
    For this class the naively approach of embedded SDC is used.
    """

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)
        u, info = gmres(
            A=self.Id0 - factor * self.A,
            b=rhs.flatten(),
            x0=u0.flatten(),
            rtol=self.lintol,
            maxiter=self.liniter,
            atol=0,
            callback=self.work_counters['gmres'],
            callback_type='legacy',
        )
        me[:] = u.reshape(me.shape)
        return me