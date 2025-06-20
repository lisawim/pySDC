import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE

class DahlquistProtheroRobinsonDAE(ProblemDAE):
    r"""
    Example implementing the combined system consisting of the Dahlquist equation
    and the Prothero-Robinson example [1]_. For :math:`\varepsilon = 0` we have the algebraic equation

    .. math::
        y'(t) = \lambda y(t) + \mu z(t),

    .. math::
        0 = -z(t) + p(t)

    for :math:`p(t) = \alpha y(t)`. The exact solution is given by
    :math:`(e^{\lambda t} + \mu \alpha t e^{\lambda t}, \alpha e^{\lambda t})`.

    Parameters
    ----------
    newton_tol : float
        Tolerance for inner solver to terminate.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts work, here, the number of right-hand side evaluations and work in inner solver
        are counted.
    """

    def __init__(self, newton_tol=1e-12, lamb=10.0, alpha=200.0, mu=1.0):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister("newton_tol", "lamb", "alpha", "mu", localVars=locals())

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
        y, z = u.diff[0], u.alg[0]
        dy = du.diff[0]

        f = self.dtype_f(self.init)
        f.diff[0] = dy - self.lamb * y - self.mu * z
        f.alg[0] = -z + self.alpha * np.exp(self.lamb * t)
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
        me.diff[0] = np.exp(self.lamb * t) * (1.0 + self.mu * self.alpha * t)
        me.alg[0] = self.alpha * np.exp(self.lamb * t)
        return me


class DahlquistProtheroRobinsonDAEConstrained(DahlquistProtheroRobinsonDAE):
    def __init__(
            self,
            nvars=2,
            newton_tol=1e-12,
            newton_maxiter=100,
            stop_at_maxiter=False,
            stop_at_nan=False,
            lamb=10.0,
            alpha=200.0,
            mu=1.0,
        ):
        """Initialization routine"""
        super().__init__(newton_tol=newton_tol, lamb=lamb, alpha=alpha)

        self._makeAttributeAndRegister("newton_maxiter", "stop_at_maxiter", "stop_at_nan", localVars=locals())

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

        y, z = u.diff[0], u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = self.lamb * y + self.mu * z
        f.alg[0] = -z + self.alpha * np.exp(self.lamb * t)
        # f.alg[0] = -z + (self.alpha * y) / (1 + self.mu * self.alpha * t)
        self.work_counters["rhs"]()
        return f

    def g(self, factor, u, t, rhs):
        y, z = u.diff[0], u.alg[0]

        g1 = y - factor * (self.lamb * y + self.mu * z) - rhs.diff[0]
        g2 = -z + self.alpha * np.exp(self.lamb * t)
        # g2 = -z + (self.alpha * y) / (1 + self.mu * self.alpha * t)

        return np.array([g1, g2])

    def dg(self, factor, t):
        dg = np.array(
            [
                [1 - factor * self.lamb, -self.mu * factor],
                [self.alpha / (1 + self.mu * self.alpha * t), -1],  # [0, -1],
            ]
        )

        return dg
    
    def dg_inv(self, factor):
        det_dg = factor * self.lamb - 1

        dg_inv = np.array(
            [
                [-1, self.mu * factor],
                [0, 1 - factor * self.lamb],
            ]
        )
        return 1 / det_dg * dg_inv

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

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = self.g(factor, u, t, rhs)

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Inverse of dg
            dg_inv = self.dg_inv(factor)
            # dg = self.dg(factor, t)

            # Newton update: u1 = u0 - g/dg
            dx = dg_inv @ g
            # dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters['newton']()
        # print(n, res)
        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class DahlquistProtheroRobinsonDAEEmbedded(DahlquistProtheroRobinsonDAEConstrained):
    def g(self, factor, u, t, rhs):
        y, z = u.diff[0], u.alg[0]

        g1 = y - factor * (self.lamb * y + self.mu * z) - rhs.diff[0]
        g2 = -factor * (-z + self.alpha * np.exp(self.lamb * t)) - rhs.alg[0]
        # g2 = -factor * (-z + (self.alpha * y) / (1 + self.mu * self.alpha * t)) - rhs.alg[0]

        return np.array([g1, g2])

    def dg(self, factor, t):
        dg = np.array(
            [
                [1 - factor * self.lamb, -self.mu * factor],
                [(-factor * self.alpha) / (1 + self.mu * self.alpha * t), factor],  # [0, factor],
            ]
        )

        return dg
    
    def dg_inv(self, factor):
        det_dg = factor - factor ** 2 * self.lamb

        dg_inv = np.array(
            [
                [factor, self.mu * factor],
                [0, 1 - factor * self.lamb],
            ]
        )
        return 1 / det_dg * dg_inv
