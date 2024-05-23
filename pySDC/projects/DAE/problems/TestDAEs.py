import numpy as np

from pySDC.core.Problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.core.Errors import ProblemError


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
        f.alg[:] = self.lamb_diff * u_diff - self.lamb_alg * u_alg
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
        me.diff[0] = np.exp(2 * self.lamb_diff * t)
        me.alg[0] = self.lamb_diff / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me


class LinearTestDAEIntegralFormulation(LinearTestDAE):
    def __init__(self, nvars=(1, 1), newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=False):
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
        f.alg[:] = self.lamb_diff * u_diff - self.lamb_alg * u_alg
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

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            y, z = u.diff[0], u.alg[0]
            f = self.eval_f(u, t)

            g = np.array([y - factor * f.diff[0] - rhs.diff[0], f.alg[0]]).flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array([[1 - self.lamb_diff * factor, -self.lamb_alg * factor], [self.lamb_diff, -self.lamb_alg]])

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g).reshape(u.shape).view(type(u))
            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        # elif np.isnan(res):
        #     self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class LinearTestDAEIntegralFormulation2(LinearTestDAEIntegralFormulation):

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
            f = self.eval_f(u, t)

            # g = np.array([y - factor * f.diff[0] - rhs.diff[0], -factor * f.alg[0] - rhs.alg[0]]).flatten()
            g = np.array([y - factor * f.diff[0] - rhs.diff[0], -factor * f.alg[0] - rhs.alg[0]])

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
        # elif np.isnan(res):
        #     self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class LinearTestDAEMinionIntegralFormulation(ptype_dae):
    def __init__(self, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True):
        super().__init__(nvars=(3, 1), newton_tol=newton_tol)
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

        u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
        u4 = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = u1 - u3 + u4
        f.diff[1] = -1e4 * u2 + (1 + 1e4) * np.exp(t)
        f.diff[2] = u1
        f.alg[0] = u1 + u2 + u4 - np.exp(t)
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

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
            u4 = u.alg[0]

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * (u1 - u3 + u4) - rhs.diff[0],
                    u2 - factor * (-1e4 * u2 + (1 + 1e4) * np.exp(t)) - rhs.diff[1],
                    u3 - factor * u1 - rhs.diff[2],
                    u1 + u2 + u4 - np.exp(t),
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, factor, -factor],
                    [0, 1 + factor * 1e4, 0, 0],
                    [-factor, 0, 1, 0],
                    [1, 1, 0, 1],
                ]
            )
            
            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.alg[0] -= dx[3]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            print(f"Nan at time {t} for eps={self.eps}")
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

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to approximate the exact solution at time t by ``SciPy`` or give initial conditions when called at :math:`t=0`.

        Parameters
        ----------
        t : float
            Current time.
        u_init : pySDC.problem.vanderpol.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Approximate exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[0] = np.cos(t)
        me.diff[1] = np.exp(t)
        me.diff[2] = np.sin(t)
        me.alg[0] = -np.cos(t)
        return me


class LinearTestDAEMinionIntegralFormulation2(LinearTestDAEMinionIntegralFormulation):

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
            u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
            u4 = u.alg[0]

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * (u1 - u3 + u4) - rhs.diff[0],
                    u2 - factor * (-1e4 * u2 + (1 + 1e4) * np.exp(t)) - rhs.diff[1],
                    u3 - factor * u1 - rhs.diff[2],
                    -factor * (u1 + u2 + u4 - np.exp(t)) - rhs.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, factor, -factor],
                    [0, 1 + factor * 1e4, 0, 0],
                    [-factor, 0, 1, 0],
                    [-factor, -factor, 0, -factor],
                ]
            )
            
            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.alg[0] -= dx[3]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            print(f"Nan at time {t} for eps={self.eps}")
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