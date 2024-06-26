import numpy as np

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.core.errors import ProblemError


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


class LinearTestDAEEmbedded(LinearTestDAE):
    r"""
    For this class the naively approach of embedded SDC is used.
    """

    def __init__(self, nvars=(1, 1), newton_tol=1e-12, newton_maxiter=50, stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

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

        u = self.dtype_u(u0)

        Id = np.identity(2)
        Id[-1, -1] = 0

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g(u), such that the solution to the nonlinear problem is a root of g

            y, z = u.diff[0], u.alg[0]
            f_diff = self.lamb_diff * y + self.lamb_alg * z
            f_alg = self.algebraicConstraints(u, t)[0]

            # g = np.array([y - factor * f.diff[0] - rhs.diff[0], -factor * f.alg[0] - rhs.alg[0]]).flatten()
            g = np.array([y - factor * f_diff - rhs.diff[0], -factor * f_alg - rhs.alg[0]])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array([[1 - self.lamb_diff * factor, -self.lamb_alg * factor], [-self.lamb_diff * factor, self.lamb_alg * factor]])

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g).reshape(u.shape).view(type(u))
            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me
