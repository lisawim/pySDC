import numpy as np

from pySDC.core.problem import WorkCounter
from pySDC.core.errors import ProblemError
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae


class LinearIndexTwoDAE(ptype_dae):
    def __init__(self, newton_tol=1e-10):
        """Initialization routine"""
        super().__init__(nvars=3, newton_tol=newton_tol)

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
            Current value of the right-hand side of f (which includes three components).
        """

        x1, x2 = u.diff[0], u.diff[1]
        dx1, dx2 = du.diff[0], du.diff[1]
        z = u.alg[0]

        # taken from Minion et al. (2011), SI-SDC-DAE paper
        f = self.dtype_f(self.init)
        f.diff[0] = dx1 - x1,
        f.diff[1] = dx2 - 2 * x1 + 1e5 * x2 - z - (1e5 + 1) * np.exp(t)
        f.alg[0] = x1 + x2
        self.work_counters['rhs']()
        return f

    def u_exact(self, t):
        """
        Routine for the exact solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing three components.
        """
        me = self.dtype_u(self.init)
        me.diff[0] = np.exp(t)
        me.diff[1] = -np.exp(t)
        me.alg[0] = -(4 + 2e5) * np.exp(t)
        return me


class LinearIndexTwoDAEIntegralFormulation(LinearIndexTwoDAE):
    def __init__(self, nvars=3, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True):
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

        x1, x2 = u.diff[0], u.diff[1]
        z = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = x1
        f.diff[1] = 2 * x1 - 1e5 * x2 + z + (1e5 + 1) * np.exp(t)
        f.alg[0] = x1 + x2
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
            x1, x2 = u.diff[0], u.diff[1]
            z = u.alg[0]
            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    x1 - factor * f.diff[0] - rhs.diff[0],
                    x2 - factor * f.diff[1] - rhs.diff[1],
                    f.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, 0],
                    [-2 * factor, 1 + 1e5 * factor, -factor],
                    [1, 1, 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

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
            # else:
            #     self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class LinearIndexTwoDAEIntegralFormulation2(LinearIndexTwoDAEIntegralFormulation):
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
            x1, x2 = u.diff[0], u.diff[1]
            z = u.alg[0]
            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    x1 - factor * f.diff[0] - rhs.diff[0],
                    x2 - factor * f.diff[1] - rhs.diff[1],
                    -factor * f.alg[0] - rhs.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, 0],
                    [-2 * factor, 1 + 1e5 * factor, -factor],
                    [-factor, -factor, 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

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
            # else:
            #     self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me